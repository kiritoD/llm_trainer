# from transformers.utils import logging
import argparse
import os
from functools import partial

import src.data.dataset as DatasetModule
import torch.distributed as dist
from src.data.tokenizer import get_tokenizer
from src.models.model import Model, ModelMapping
from src.peft_utils.convert import PEFT
from src.trainer.trainer import (
    TrainerMapping,
    CausalFtTrainer,
    SequenceClassificationTrainer,
    LLMCallback,
)
from src.utils import (
    get_logger,
    is_deepspeed_zero3,
    parse_arguments,
    rank_zero_info,
)

logger = get_logger("LLM_TRAINER")
rank_zero_info = partial(rank_zero_info, logger=logger)

train_parser = argparse.ArgumentParser(
    prog="PEFT_LLM_TRAIN", description="train for llm"
)
train_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="a config file path which contains all arguments for train",
)


def train(
    train_data_params: dict,
    tokenizer_params: dict,
    model_params: dict,
    trainer_params: dict,
    peft_params: dict,
):

    # data load
    rank_zero_info("start to preprocess the dataset ...")
    dataset_class_name = train_data_params.pop("dataset_class_name", None)
    assert (
        dataset_class_name
    ), f"please set the key `dataset_name` in parameter yaml file"
    dataset_class = getattr(DatasetModule, dataset_class_name, None)
    assert dataset_class, f"don't have this class for dataset!"
    train_data_params.update(model_params)
    train_data_params.update(trainer_params)
    dataset_t = dataset_class(train_data_params)

    #  get the model
    rank_zero_info("start to load the model ...")
    # if not trainer_params.get("resume_from_checkpoint", None):
    model = Model(model_params)
    model_type = model_params["model_type"]
    model_ins = getattr(model, ModelMapping.get(model_type, "get_causal_lm"))(
        num_labels=(
            None
            if model_params["model_type"] != "SequenceClassification"
            else dataset_t.num_labels
        ),
        finetuning_task=(
            None
            if model_params["model_type"] != "SequenceClassification"
            else train_data_params["train_file"]
        ),
    )

    # tokernizer init
    rank_zero_info("start to load the tokenizer ...")
    tokenizer_params.update(model_params)
    tokenizer = get_tokenizer(tokenizer_params)

    # data preprocess
    dataset_t.post_process(
        model=model_ins, tokenizer=tokenizer, config=model_ins.config
    )
    # get train_dataset and the collator
    train_size = train_data_params.get("train_size", -1)
    train_dataset = dataset_t.get_train_dataset(train_size)

    # get eval_dataset and the collator
    eval_dataset = None
    data_collator_eval = None
    if train_data_params.get("eval_file", None):
        eval_size = train_data_params.get("eval_size", -1)
        eval_dataset = dataset_t.get_eval_dataset(eval_size)

    rank_zero_info("partial the collate function ...")
    if model_params["model_type"] == "Causal":
        data_collator = partial(
            dataset_t.collate_fn,
            tokenizer=tokenizer,
            max_len=tokenizer_params["max_len"],
            batch_pad_truncate=tokenizer_params["batch_pad_truncate"],
            ignore_q=tokenizer_params["ignore_q"],
        )
    else:
        data_collator = dataset_t.collate_fn
    #  model preprocess
    try:
        if len(tokenizer) != model_ins.config.vocab_size:
            model_ins.resize_token_embeddings(len(tokenizer))
    except:
        ...

    # peft process
    peft_type = None
    if model_params["peft"] and not trainer_params.get("resume_from_checkpoint", None):
        peft_type = peft_params["peft_type"]
        rank_zero_info("start to proces the model by PEFT ...")
        peft = PEFT(peft_params)
        model_ins = peft.get_peft_model(model_ins)

    # for adamw_torch_fused
    if (
        not is_deepspeed_zero3(trainer_params)
        and not trainer_params.get("resume_from_checkpoint", None)
        and model_params["model_type"] == "Causal"
    ):
        model_ins.to(f"cuda:{dist.get_rank()}")
    # if not trainer_params.get("resume_from_checkpoint", None):
    Model.count_pramameter(model_ins, verbose=True)
    # trainer init
    rank_zero_info("start to init the trainer ...")
    optimizers = (None, None)
    resume_from_checkpoint = trainer_params.get("resume_from_checkpoint", False)
    # trainer combine all modules
    TrainerClass = TrainerMapping.get(model_params["model_type"], CausalFtTrainer)
    trainer = TrainerClass(
        params=trainer_params,
        model=model_ins,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
        compute_metrics=dataset_t.metric,
        optimizers=optimizers,
    )

    rank_zero_info("start to train")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save the model
    if not model_params["peft"]:
        trainer.save_model()
    # log the train metrics
    if model_params["model_type"] == "SequenceClassification":
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        if trainer_params.get("do_eval", False):
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [train_data_params.get("eval_file")]
            eval_datasets = [eval_dataset]
            if train_data_params.get("eval_file") == "mnli":
                tasks.append("mnli-mm")
                valid_mm_dataset = dataset_t.get_eval_dataset(eval_size)
                eval_datasets.append(valid_mm_dataset)
                combined = {}

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = eval_size if eval_size != -1 else len(eval_dataset)
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

                if task == "mnli-mm":
                    metrics = {k + "_mm": v for k, v in metrics.items()}
                if task is not None and "mnli" in task:
                    combined.update(metrics)

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics(
                    "eval", combined if task is not None and "mnli" in task else metrics
                )

    dist.barrier()
    rank_zero_info("training over")


def main(config_path: str):
    # ddp init
    logger.info("start to init process group ...")
    dist.init_process_group(
        "nccl",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    # parse arguments
    config = parse_arguments(config_path)
    # parse config data
    train_data_params = config["data-params"]
    tokenizer_params = config["tokenizer-params"]
    model_params = config["model-params"]
    trainer_params = config["trainer-params"]
    peft_params = config["peft-params"]

    #  start to train
    train(
        train_data_params,
        tokenizer_params,
        model_params,
        trainer_params,
        peft_params,
    )


if __name__ == "__main__":
    train_args = train_parser.parse_args()
    main(train_args.config_path)

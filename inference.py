# from transformers.utils import logging
import argparse
import os
from functools import partial

import src.data.dataset as DatasetModule
import torch
import torch.distributed as dist
from src.data.tokenizer import get_tokenizer
from src.models.model import Model
from src.peft_utils.convert import PEFT
from src.trainer.trainer import CausalFtTrainer
from src.utils import get_logger, parse_arguments, rank_zero_info

logger = get_logger("LLM_inference")
rank_zero_info = partial(rank_zero_info, logger=logger)

test_parser = argparse.ArgumentParser(prog="PEFT_LLM_inference", description="inference for llm")
test_parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="a config file path which contains all arguments for inference",
)


def predict(
    test_data_params: dict,
    tokenizer_params: dict,
    model_params: dict,
    trainer_params: dict,
):
    # data preprocess
    rank_zero_info("start to preprocess the test dataset ...")
    test_dataset_class_name = test_data_params.pop("dataset_class_name", None)
    assert test_dataset_class_name, f"please set the key `dataset_name` in parameter yaml file"
    test_dataset_class = getattr(DatasetModule, test_dataset_class_name, None)
    assert test_dataset_class, f"don't have this class for test dataset!"
    test_dataset: DatasetModule.CommonDataset = test_dataset_class(test_data_params)
    test_size = test_data_params.get("test_size", -1)
    dataset = test_dataset.get_test_dataset(test_size)

    # tokernizer init
    rank_zero_info("start to load the tokenizer ...")
    tokenizer = get_tokenizer(tokenizer_params)
    if "llama" in tokenizer_params["pretrain_tokenizer_path"].lower():
        # if llama model, for batch inference, should set pad_token_id = 0
        tokenizer.pad_token_id = 0

    # get the collator
    rank_zero_info("partial the inference collate function ...")
    data_collator = partial(
        test_dataset.collate_fn,
        tokenizer=tokenizer,
        max_len=tokenizer_params["max_len"],
        mode="test",
    )

    #  get the model
    load_weight_after_peft = model_params["load_weight_after_peft"]
    if load_weight_after_peft:
        rank_zero_info("load weights after peft")
    rank_zero_info("start to load the model ...")
    model = Model(model_params)
    causal_model = model.get_causal_lm(load_weight_after_peft=load_weight_after_peft)

    #  model preprocess
    try:
        if len(tokenizer) != causal_model.config.vocab_size:
            causal_model.resize_token_embeddings(len(tokenizer))
    except:
        ...

    # peft process
    if model_params["peft"]:
        rank_zero_info("start to use peft process the model ...")
        causal_model = PEFT.from_pretrained(
            causal_model,
            model_params["peft_model_path"],
            weights_merge=model_params["weights_merge"],
        )

    causal_model = causal_model.to(torch.float16)
    # load model weights
    if load_weight_after_peft:
        rank_zero_info("load weights after peft")
        state_dict = torch.load(
            model_params["checkpoint_model_path"],
            map_location=f"cuda:{dist.get_rank()}",
        )
        causal_model.load_state_dict(state_dict, strict=False)
        del state_dict

    model.count_pramameter(causal_model)
    # test init
    rank_zero_info("start to init the inference ...")

    # test combine all modules
    predictor = CausalFtTrainer(
        params=trainer_params,
        model=causal_model,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    rank_zero_info("start to inference")
    predictor.predict(dataset)

    # ----------------------------

    # ----------------------------

    dist.barrier()
    rank_zero_info("inference over")


def main(config_path: str):
    # TODO: support V100
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
    test_data_params = config["data-params"]
    tokenizer_params = config["tokenizer-params"]
    model_params = config["model-params"]
    trainer_params = config["trainer-params"]

    test_model_weights_paths = []
    model_wegiht_path = None
    key_name = None
    if model_params["peft"]:
        key_name = "peft_model_path"
        model_wegiht_path = model_params[key_name]
    else:
        key_name = "checkpoint_model_path"
        model_wegiht_path = model_params[key_name]

    if isinstance(model_wegiht_path, str):
        test_model_weights_paths.append(model_wegiht_path)
    elif isinstance(model_wegiht_path, list):
        for model_wegiht_path_ in model_wegiht_path:
            test_model_weights_paths.append(model_wegiht_path_)
    else:
        test_model_weights_paths.append(model_params["pretrain_model_path"])

    predict_file_name_origin = trainer_params["prediction_file_name"]
    for model_wegiht_path_ in test_model_weights_paths:
        rank_zero_info(f"start predict based on {model_wegiht_path_}")
        model_params[key_name] = model_wegiht_path_
        output_name = model_wegiht_path_
        if os.path.isfile(output_name):
            output_name = os.path.dirname(model_wegiht_path_)
        trainer_params["prediction_file_name"] = os.path.join(output_name, predict_file_name_origin)
        # if dist.get_rank() == 0:
        #     if key_name == "checkpoint_model_path":
        #         pytorch_model_path = os.path.join(output_name, "pytorch_model.bin")
        #         if not os.path.exists(pytorch_model_path):
        #             os.system(
        #                 f"cd {output_name} && python zero_to_fp32.py ./ ./pytorch_model.bin"
        #             )
        # dist.barrier()
        rank_zero_info(f"will output the results to {trainer_params['prediction_file_name']}")
        predict(
            test_data_params.copy(),
            tokenizer_params.copy(),
            model_params.copy(),
            trainer_params.copy(),
        )
        rank_zero_info(f"output the results to `{trainer_params['prediction_file_name']}` successfully~")
        dist.barrier()


if __name__ == "__main__":
    test_args = test_parser.parse_args()
    main(test_args.config_path)

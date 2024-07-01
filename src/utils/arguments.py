from cgi import test
import os
import time
from dataclasses import dataclass, field

import torch.distributed as dist
from prettytable import PrettyTable

from .auxiliary import load_yaml
from .logging import get_logger

# from transformers import AutoConfig

logger = get_logger("Arguments")


def setup_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


def hash_time():
    time_str = time.strftime(r"%Y_%m_%d_%H_%M_%S")
    return time_str


def output_path_preprocess(trainer_params):
    logger.info(time.strftime(r"%Y_%m_%d_%H_%M_%s"))
    if not trainer_params.get("do_predict", False):
        if os.path.exists(trainer_params["output_dir"]):
            predition_file_old_name = trainer_params["prediction_file_name"]

            trainer_params["prediction_file_name"] = "_".join(
                [
                    trainer_params["prediction_file_name"].split(".")[0],
                    hash_time()
                    + "."
                    + trainer_params["prediction_file_name"].split(".")[-1],
                ]
            )
            logger.warn(
                f"the prediction_file: `{predition_file_old_name}` exists, the new prediction file name: `{trainer_params['prediction_file_name']}`"
            )

    if os.path.exists(trainer_params["output_dir"]) and not trainer_params.get(
        "resume_from_checkpoint"
    ):
        old_output_dir = trainer_params["output_dir"]
        if len(os.listdir(old_output_dir)) > 0:
            trainer_params["output_dir"] = "_".join(
                [
                    trainer_params["output_dir"],
                    hash_time(),
                ]
            )
            logger.warn(
                f"the output directory: `{old_output_dir}` exists, the new output dir is: `{trainer_params['output_dir']}`~"
            )
    auto_resume = (
        trainer_params.pop("auto_resume", False)
        if "auto_resume" in trainer_params
        else False
    )
    auto_resume_save_steps = (
        trainer_params.pop("auto_resume_save_steps", 200)
        if "auto_resume_save_steps" in trainer_params
        else 200
    )
    if auto_resume:
        trainer_params["save_strategy"] = "steps"
        trainer_params["save_steps"] = auto_resume_save_steps
        trainer_params["save_total_limit"] = 1
        if os.path.exists(trainer_params["output_dir"]):
            has_pth = False
            for root, dirs, files in os.walk(trainer_params["output_dir"]):
                for file in files:
                    if file.endswith(".pth"):
                        has_pth = True
            # 如果输出路径存在但是没有ckpt，resume失效
            trainer_params["resume_from_checkpoint"] = has_pth
        elif not os.path.exists(trainer_params["output_dir"]):
            # 如果输出路径不存在，resume失效
            trainer_params["resume_from_checkpoint"] = False
        else:
            ...

@dataclass
class ModelArguments:
    # the target AutoModelForxxx.from_pretrained, the following arguments are the extral arguments that are used for our training
    peft: bool = field(
        default=False,
        metadata={"help": ("if train peft model")},
    )
    model_type: str = field(
        default="Causal",
        metadata={"help": ("model type. Possible choices are the following: 'Causal', 'SequenceClassification' ")},
    )

@dataclass
class DataArguments:
    dataset_class_name: str = field(
        default="SIQADataset",
        metadata={"help": ("target dataset")},
    )
    train_file: str = field(
        default="SIQADataset",
        metadata={"help": ("train dataset")},
    )
    train_size: int = field(
        default=10000,
        metadata={"help": ("train dataset size")},
    )
    eval_file: str = field(
        default="social_i_qa",
        metadata={"help": ("eval dataset")},
    )
    eval_size: int = field(
        default=1000,
        metadata={"help": ("eval dataset size")},
    )
    test_file: str = field(
        default="social_i_qa",
        metadata={"help": ("test dataset")},
    )
    test_size: int = field(
        default=1000,
        metadata={"help": ("test dataset size")},
    )
    

def parse_arguments(config_path: str):
    config_dict = load_yaml(config_path)
    wandb_project_name = config_dict["trainer-params"].get(
        "wandb_project_name", None
    )
    if config_dict["model-params"].get("model_type", "Causal") == "Causal":
        output_path_preprocess(config_dict["trainer-params"])
    if wandb_project_name:
        os.environ["WANDB_PROJECT"] = wandb_project_name
    seed = config_dict["trainer-params"].get("seed", None)
    if seed:
        setup_seed(seed)
    # pass peft value to trainer param
    config_dict["trainer-params"]["peft"] = config_dict["model-params"]["peft"]
    # should optimize this part for extra arguments
    model_extra_args = ModelArguments.__dataclass_fields__
    model_extra_args_keys = set(model_extra_args.keys())
    for extra_arg in model_extra_args_keys:
        if extra_arg not in config_dict["model-params"]:
            config_dict["model-params"][extra_arg] = model_extra_args[extra_arg].default
    
    for key, args in config_dict.items():
        arg_table = PrettyTable()
        arg_table.field_names = ["key", "value"]
        
        if isinstance(args, dict):
            arg_table.add_rows(list(args.items()))
            if key == "peft-params" and not config_dict["model-params"]["peft"]:
                continue
            if dist.get_rank() == 0:
                logger.info(f"{key}:\n" + str(arg_table))

    time.sleep(2)
    return config_dict

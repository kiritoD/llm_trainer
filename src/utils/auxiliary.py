import json
from typing import Any, Dict
import jsonlines
import torch
import yaml
import pandas as pd


# file load utils
def is_deepspeed_zero3(trainer_params):
    ds_config = trainer_params.get("deepspeed", None)
    is_zero3 = False
    if ds_config:
        if isinstance(ds_config, str):
            with open(ds_config, "r") as f:
                ds_config = json.load(f)
            if ds_config["zero_optimization"]["stage"] == 3:
                is_zero3 = True
        elif isinstance(ds_config, dict):
            try:
                if ds_config["zero_optimization"]["stage"] == 3:
                    is_zero3 = True
            except:
                ...
    return is_zero3


def load_yaml(path: str) -> Dict[str, Any]:
    """load yaml file, convert yaml to dict

    Parameters
    ----------
    path : str
        the yaml file path

    Returns
    -------
    Dict[str, Any]
        dict data
    """

    with open(path, "r") as yaml_file:
        data = yaml.full_load(yaml_file)

    return data


def jsonfile2yamlfile(json_path: str, yaml_path: str):
    json_dict = load_json(json_path)
    json2yaml(yaml_path, json_dict)


def json2yaml(path: str, data: Dict[str, Any]):
    """output json data to yaml file

    Parameters
    ----------
    path : str
        the target output yaml file path
    data : Dict[str, Any]
        the json data
    """
    with open(path, "w", encoding="utf-8") as writer:
        yaml.dump(data, writer)


def jsonl2csv(path: str):
    """convert jsonl file to csv file

    Parameters
    ----------
    path : str
        the original jsonl file
    """
    jr = jsonlines.Reader(open(path, "r"))
    path_output = path.replace(".jsonl", ".csv")
    json_arr = []
    for line in jr:
        json_arr.append(line)

    df = pd.DataFrame(json_arr)
    df.to_csv(path_output)


def load_json(path: str) -> Dict[str, Any]:
    """load json file, convert json to dict

    Parameters
    ----------
    path : str
        the json file path

    Returns
    -------
    Dict[str, Any]
        dict data
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)

    return data


def truncate_pad_for_input(
    input_ids,
    if_prefix: bool = True,
    if_postfix: bool = True,
    pad_id: int = 3,
):
    """try to truncate the input_ids for the model train and inference

    Parameters
    ----------
    input_ids : torch.Tensor
        _description_
    if_prefix : bool, optional
        _description_, by default True
    if_postfix : bool, optional
        _description_, by default True
    pad_id : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """
    output_trs = input_ids.clone().detach().transpose(1, 0)
    length_y, length_x = output_trs.size()[:2]
    start_y, end_y = -1, -1

    for index in range(length_y):
        if start_y == -1 and output_trs[index].sum() != pad_id * length_x:
            start_y = index
            continue
        if start_y != -1 and output_trs[index].sum() == pad_id * length_x:
            end_y = index
            break

    if not if_prefix:
        start_y = 0
    if not if_postfix or end_y == -1:
        end_y = length_y
    return start_y, end_y


def merge_jsonl(path_re, path):
    import glob

    with open(path, "w") as writer:
        for file_name in glob.glob(path_re):
            with open(file_name, "r") as f:
                for line in f:
                    writer.write(line)


def model_linear_postfix(model):
    """
    get the postfix of nn.Linear layer for lora
    """
    name_arr = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name_arr.append(name)
    linear_map = list(map(lambda x: x.split(".")[-1], name_arr))
    name_arr_new = list(set(linear_map))
    linear_map_dict = {key: 0 for key in name_arr_new}
    for item in linear_map:
        linear_map_dict[item] += 1
    return name_arr_new, linear_map_dict


if __name__ == "__main__":
    path_re = "./model_output/external_platform_c2_model_ft_lr_2e-5/checkpoint-2084/result_remote_prob_*.jsonl"
    path = "./model_output/external_platform_c2_model_ft_lr_2e-5/checkpoint-2084/result_remote_prob.jsonl"
    merge_jsonl(path_re, path)

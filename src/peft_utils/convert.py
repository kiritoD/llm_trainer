import os
from functools import partial
from typing import Any, Dict, Union

import torch

# from ..peft_a_lora import (
#     AdaLoraConfig,
#     LoraConfig,
#     NASLoraConfig,
#     PeftModel,
#     PrefixTuningConfig,
#     PromptEncoderConfig,
#     PromptTuningConfig,
#     get_peft_config,
#     get_peft_model,
# )
from ..peft import (
    AdaLoraConfig,
    LoraConfig,
    Lora_AConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_peft_config,
    get_peft_model,
    NASLoraConfig,
    VeraConfig,
    LoCPConfig,
    HSSAConfig,
)

# from peft import (
#     PeftModel,
#     AdaLoraConfig,
#     PrefixTuningConfig,
#     PromptEncoderConfig,
#     PromptTuningConfig,
#     get_peft_config,
#     get_peft_model,
# )
from ..utils.auxiliary import json2yaml, load_json, load_yaml
from ..utils.logging import get_logger, rank_zero_info

logger = get_logger("PEFT")

# PEFT_TYPE_2_CONFIG_MAPPING = {
#     "lora": LoraConfig,
#     "p_tuning": PromptEncoderConfig,
#     "adalora": AdaLoraConfig,
#     "naslora": NASLoraConfig,
#     "prefix_tuning": PrefixTuningConfig,
#     "prompt_tuning": PromptTuningConfig,
# }

PEFT_TYPE_2_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "p_tuning": PromptEncoderConfig,
    "adalora": AdaLoraConfig,
    "lora_a": Lora_AConfig,
    "prefix_tuning": PrefixTuningConfig,
    "prompt_tuning": PromptTuningConfig,
    "naslora": NASLoraConfig,
    "vera": VeraConfig,
    "locp": LoCPConfig,
    "hssa": HSSAConfig,
}


class PEFT:
    rank_zero_info = partial(rank_zero_info, logger=logger)

    def __init__(self, config_path_or_data: Union[str, dict]) -> None:
        if isinstance(config_path_or_data, str):
            self.config_path = config_path_or_data
            self.config = self._get_config(config_path_or_data)
        else:
            self.config = config_path_or_data
        self.raw_config = self.config.copy()
        self.peft_type = self.config.pop("peft_type", None)
        assert self.peft_type, f"{self.config_path} shoud set the `peft_type` key"
        self.peft_type: str = self.peft_type.lower()

    @classmethod
    def _get_config(cls, path: str) -> Dict[str, Any]:
        if path.endswith("json"):
            return load_json(path)
        if path.endswith("yaml") or path.endswith("yml"):
            return load_yaml(path)

    def get_peft_model(
        self, model: torch.nn.Module, normal: bool = True
    ) -> torch.nn.Module:
        """preprocess the peft config and customize your own model

        Parameters
        ----------
        model : torch.nn.Module
            the model which is from transformers
        normal : bool, optional
            if normal, use the peft module to process your model, otherwise, customize your model by some special technique, by default True

        Returns
        -------
        torch.nn.Module
            peft model
        """
        peft_config = PEFT_TYPE_2_CONFIG_MAPPING[self.peft_type](**self.config)
        # Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping the model weights fixed.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        peft_config_strs = [
            item[0] + ": " + str(item[1]) + " | " for item in self.config.items()
        ]

        self.rank_zero_info(
            f"| PEFT type: {self.peft_type} | {''.join(peft_config_strs)}"
        )

        if normal:
            peft_model = get_peft_model(model, peft_config)
            self.print_trainable_parameters(peft_model)

        return peft_model

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params_total = 0
        trainable_params_target = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params_total += num_params
                if "classifier" not in _:
                    trainable_params_target += num_params
        print(
            f"total trainable params: {trainable_params_total} || target trainable params: {trainable_params_target} || all params: {all_param} || total trainable%: {100 * trainable_params_total / all_param} || target trainable%: {100 * trainable_params_target / all_param}"
        )

    def output_config(self, path: str):
        self.rank_zero_info(f"output the config into the path: {path}")
        json2yaml(path, self.raw_config)

    @classmethod
    def weights_merge(cls, peft_model):
        if getattr(peft_model, "merge_adapter"):
            cls.rank_zero_info("start to merge adapter into model")
            # TODO: problem about the precision
            peft_model.merge_adapter()
        else:
            cls.rank_zero_info(
                "the model not implement `merge_adapter` method, merge adapter failed"
            )

    @classmethod
    def from_pretrained(
        cls, model, peft_pretrained_model_path, weights_merge: bool = False
    ) -> torch.nn.Module:
        """load pretrained peft model from `peft_pretrained_model_path`

        Parameters
        ----------
        model : torch.nn.Module
            the model which is from transformers
        weights_merge: bool
            whether merge adapter weights or not, if merge weights, it may lead to a poor model performance

        Returns
        -------
        torch.nn.Module
            peft model
        """
        cls.rank_zero_info(
            f"start to load the pretrained peft adapter from {peft_pretrained_model_path}"
        )
        config_path = os.path.join(peft_pretrained_model_path, "adapter_config.json")
        config_dict = cls._get_config(config_path)
        train_inference_mode = False
        peft_type = None
        if config_dict.get("inference_mode", None) == False:
            cls.rank_zero_info(
                f"the model will continue finetune from {peft_pretrained_model_path}"
            )
            train_inference_mode = True
            peft_type = config_dict.get("peft_type", None)

        peft_model_pretrained = PeftModel.from_pretrained(
            model, peft_pretrained_model_path
        )
        if weights_merge:
            cls.weights_merge(peft_model_pretrained)
        if train_inference_mode and peft_type:
            peft_type: str = peft_type.lower()
            for name, param in peft_model_pretrained.named_parameters():
                if peft_type in name:
                    param.requires_grad = True
        if hasattr(model, "enable_input_require_grads") and train_inference_mode:
            model.enable_input_require_grads()
        peft_model_pretrained.print_trainable_parameters()
        cls.rank_zero_info(f"load the pretrained peft adapter successfully!")
        if not peft_type:
            return peft_model_pretrained
        else:
            return peft_model_pretrained, peft_type

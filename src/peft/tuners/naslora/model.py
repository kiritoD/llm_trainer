# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import re
import warnings
from dataclasses import asdict
from enum import Enum
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from ....utils.auxiliary import load_json
from ...import_utils import is_bnb_available
from ...utils import (
    TRANSFORMERS_MODELS_TO_NASLORA_TARGET_MODULES_MAPPING,
    _freeze_adapter,
    _get_submodules,
)
from .layer import NASLoraLayer, Linear, Linear8bitLt


if is_bnb_available():
    import bitsandbytes as bnb




class NASLoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (NASLora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`NASLoraConfig`]): The configuration of the NASLora model.

    Returns:
        `torch.nn.Module`: The NASLora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, NASLoraConfig
        >>> from peft import NASLoraModel, NASLoraConfig

        >>> config = NASLoraConfig(
        ...     peft_type="NASLORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     nas_search_space="common.yml"
        ...     target_modules=["q", "v"],
        ...     naslora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> naslora_model = NASLoraModel(config, model)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`NASLoraConfig`]): The configuration of the NASLora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self._get_naslora_mask(adapter_name)
        self.prune = False
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        self.adapter_name = adapter_name
        self.adapter_masks = {}
        self.naslora_weights = {}

    def _get_naslora_mask(self, adapter_name):
        if self.peft_config[adapter_name].naslora_mask and isinstance(
            self.peft_config[adapter_name].naslora_mask, str
        ):
            self.peft_config[adapter_name].naslora_mask = load_json(self.peft_config[adapter_name].naslora_mask)

    @property
    def arch_parameters(self):
        _arch_parameters = []
        for name, params in self.model.named_parameters():
            if "naslora_weights" in name:
                _arch_parameters.append(params)
        _arch_parameters = _arch_parameters
        return _arch_parameters

    @property
    def weights_parameters(self):
        _weight_parameters = []
        for name, params in self.model.named_parameters():
            if "naslora_module" in name:
                _weight_parameters.append(params)
        return _weight_parameters

    def save_top_n_adapters(self):
        """save topN naslora layers in peftmodel"""
        if not self.prune:
            for key, _ in self.model.named_modules():
                if "naslora" in key:
                    parent, target, target_name = _get_submodules(self.model, key)
                    if isinstance(parent, NASLoraLayer):
                        parent.save_top_n_adapters(
                            self.peft_config[self.adapter_name].top_n,
                            self.peft_config[self.adapter_name].naslora_dropout,
                        )
                        key = ".".join(key.split(".")[:-1])
                        if key not in self.adapter_masks:
                            self.adapter_masks[key] = parent.naslora_mask[self.adapter_name].cpu().numpy().tolist()
                            self.naslora_weights[key] = parent.naslora_weights_backup[self.adapter_name]

            self.peft_config[self.adapter_name].naslora_mask = self.adapter_masks
            self.prune = True

    def save_adapter_weights(self, output_dir):
        weights_file_name = os.path.join(output_dir, "adapter_weights.json")
        mask_file_name = os.path.join(output_dir, "adapter_masks.json")

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(weights_file_name):
            with open(weights_file_name, "w") as f:
                json.dump(self.naslora_weights, f)
            with open(mask_file_name, "w") as f:
                json.dump(self.adapter_masks, f)

        ...

    def reg_loss(self):
        _arch_parameters = self.arch_parameters
        if len(_arch_parameters) > 0 and _arch_parameters[0].requires_grad == True:
            _arch_parameters_stack = torch.stack(_arch_parameters)
            neg_loss = torch.logsumexp(torch.abs(_arch_parameters_stack), dim=-1)
            # neg_loss = torch.logsumexp(_arch_parameters_stack, dim=-1)
            # neg_loss = torch.pow(torch.abs(_arch_parameters_stack - 1), 2)
            # aux_loss = torch.sum(neg_loss) * 1e-3
            # aux_loss = torch.mean(neg_loss) * 1e-3
            aux_loss = torch.mean(neg_loss)
            return aux_loss
        return 0
        ...

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            config = self._prepare_naslora_config(config, self.model.config.to_dict())
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "NASLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_naslora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        naslora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use NASLora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "nas_search_space": naslora_config.nas_search_space,
            "naslora_dropout": naslora_config.naslora_dropout,
            "fan_in_fan_out": naslora_config.fan_in_fan_out,
            "init_naslora_weights": naslora_config.init_naslora_weights,
            "naslora_mask": naslora_config.naslora_mask,
            "reset_mode": naslora_config.reset_mode,
            "norm_eps": naslora_config.norm_eps,
            "layernorm": naslora_config.layernorm,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(naslora_config.target_modules, str):
                target_module_found = re.fullmatch(naslora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in naslora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                # for naslora mask
                if naslora_config.naslora_mask and key in naslora_config.naslora_mask:
                    kwargs["naslora_mask"] = naslora_config.naslora_mask[key]
                if isinstance(target, NASLoraLayer):
                    target.update_layer(
                        adapter_name,
                        naslora_config.nas_search_space,
                        naslora_config.naslora_dropout,
                        naslora_config.init_naslora_weights,
                        kwargs["naslora_mask"],
                        kwargs["norm_eps"],
                        kwargs["layernorm"],
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = Linear8bitLt(
                            adapter_name,
                            target.in_features,
                            target.out_features,
                            bias=bias,
                            **kwargs,
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = (
                                target.in_features,
                                target.out_features,
                            )
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = naslora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = naslora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = Linear(
                            adapter_name,
                            in_features,
                            out_features,
                            bias=bias,
                            **kwargs,
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {naslora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "naslora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, NASLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, NASLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, NASLoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, NASLoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_naslora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_NASLORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_NASLORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if self.config.model_type == "gpt2":
            raise ValueError("GPT2 models are not supported for merging NASLORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge NASLORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "naslora" not in key]
        for key in key_list:
            parent, target, target_name = _get_submodules(self.model, key)
            if isinstance(target, NASLoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)
        return self.model

    def lora_components_freezer(self, bias="none", freeze_mode=0) -> None:
        r"""
        Provides a method to decide which part of parameters to freeze or unfreeze

        Arguments:
            bias: str, 'none', 'all', or 'naslora_only'
            mode: LoRA weights | NASLoRA weights
                0 -  unfreezed | unfreezed
                1 -    freezed | unfreezed
                2 -  unfreezed | freezed
                3 -    freezed | freezed
        """

        def freeze_unfreeze_by_name(name, freeze):
            for n, p in self.model.named_parameters():
                if name in n:
                    if freeze:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
            if bias == "none":
                return
            elif bias == "all":
                for n, p in self.model.named_parameters():
                    if (name in n) and ("bias" in n):
                        if freeze:
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
            elif bias == "naslora_only":
                for m in self.model.modules():
                    if isinstance(m, NASLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        for n, p in m.named_parameters():
                            if freeze:
                                p.requires_grad = False
                            else:
                                p.requires_grad = True
            else:
                raise NotImplementedError

        if freeze_mode == 0:
            freeze_unfreeze_by_name("naslora_module", freeze=False)
            freeze_unfreeze_by_name("naslora_layernorms", freeze=False)
            freeze_unfreeze_by_name("naslora_weights", freeze=False)
        elif freeze_mode == 1:
            freeze_unfreeze_by_name("naslora_module", freeze=True)
            freeze_unfreeze_by_name("naslora_layernorms", freeze=True)
            freeze_unfreeze_by_name("naslora_weights", freeze=False)
        elif freeze_mode == 2:
            freeze_unfreeze_by_name("naslora_module", freeze=False)
            freeze_unfreeze_by_name("naslora_layernorms", freeze=False)
            freeze_unfreeze_by_name("naslora_weights", freeze=True)
        elif freeze_mode == 3:
            freeze_unfreeze_by_name("naslora_module", freeze=True)
            freeze_unfreeze_by_name("naslora_layernorms", freeze=True)
            freeze_unfreeze_by_name("naslora_weights", freeze=True)
        else:
            raise NotImplementedError


# had to adapt it for `naslora_only` to work
def mark_only_naslora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "naslora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "naslora_only":
        for m in model.modules():
            if isinstance(m, NASLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError



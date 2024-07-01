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
import math
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from sqlite3 import adapters
from typing import List, Optional, Union

import numpy as np
import src.peft.tuners.lora as LoRA
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ...utils.auxiliary import load_json, load_yaml
from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_NASLORA_TARGET_MODULES_MAPPING,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import ACTIVATE_FN_MAPPING, LoraLayer

if is_bnb_available():
    import bitsandbytes as bnb

PEFT_DROPOUT_MAPPING = dict(lora="lora_dropout")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@dataclass
class NASLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`NASLoraModel`].

    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply NASLora to.
        naslora_dropout (`float`): The dropout probability for NASLora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for NASLora. Can be 'none', 'all' or 'naslora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    nas_search_space: str = field(
        default="./config/search_space/common.yml",
        metadata={"help": "the nas search space yaml file"},
    )
    naslora_mask: str = field(
        default=None,
        metadata={"help": "used for mask lora variants input"},
    )
    naslora_weights_lr: float = field(
        default=None,
        metadata={"help": "choice lr"},
    )
    reg_loss: bool = field(
        default=False,
        metadata={"help": "whether use reg loss"},
    )
    search_strategy: str = field(
        default="epoch",
        metadata={"help": "epoch or step"},
    )
    search_policy: str = field(
        default="None",
        metadata={"help": "assign policy strategy for NASLoRA components"},
    )
    search_lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "assign learning rate scheduler strategy for NASLoRA components, e.g., steplr or consine"},
    )
    policy_params: dict = field(
        default=None,
        metadata={"help": "`search_policy` arguments"},
    )
    search_step: int = field(
        default=1,
        metadata={"help": "how long to search for googl architecture"},
    )
    step_size: int = field(
        default=5,
        metadata={"help": "how long to next step"},
    )
    gamma: float = field(
        default=0.9,
        metadata={"help": "the learning rate about the operator weights"},
    )
    reset_mode: int = field(
        default=2,
        metadata={"help": "0 not resest, 1 reset B, 2 reset all module"},
    )
    layernorm: bool = field(
        default=False,
        metadata={"help": "whether use layernorm during search stage"},
    )
    operator_lr_rating: float = field(
        default=0.02,
        metadata={"help": "the learning rate about the operator weights"},
    )
    top_n: int = field(default=1, metadata={"help": "save `top_n` adapters"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with NASLora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    naslora_dropout: float = field(default=0.0, metadata={"help": "NASLora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for NASLora. Can be 'none', 'all' or 'naslora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_naslora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the NASLora layers."},
    )
    norm_eps: float = field(default=1e-5, metadata={"help": "NASLora layernorm eps"})

    def __post_init__(self):
        self.peft_type = PeftType.NASLORA


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


class NASLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.nas_search_space = {}
        self.activate_fn = {}
        self.naslora_dropout = nn.ModuleDict({})
        self.naslora_module = nn.ModuleDict({})
        self.naslora_weights = nn.ParameterDict({})
        self.naslora_weights_backup = {}
        self.naslora_mask = {}
        self.nas_search_space_dict = {}
        self.naslora_layernorms = nn.ModuleDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def _get_nas_seach_space(self, nas_search_space_config_path: str):
        return load_yaml(nas_search_space_config_path)

    def _get_sub_module(self, key: str, module_config: dict, init_lora_weights):
        if not getattr(self, "lora", None):
            self.lora = LoraLayer(self.in_features, self.out_features)
        if len(module_config) != 0:
            self.lora.update_layer(
                adapter_name=key,
                init_lora_weights=init_lora_weights,
                to_device=False,
                **module_config,
            )
            if module_config["d_parts"] > 1:
                return self.lora.lora_AB[key]
            else:
                return nn.ModuleList([self.lora.lora_A[key], self.lora.lora_B[key]])
        else:
            return nn.Identity()
        ...

    def update_layer(
        self,
        adapter_name,
        nas_search_space,
        naslora_dropout,
        init_naslora_weights,
        naslora_mask,
        norm_eps,
        layernorm,
    ):
        self.nas_search_space[adapter_name] = nas_search_space
        self.activate_fn[adapter_name] = []
        self.nas_search_space_dict[adapter_name] = self._get_nas_seach_space(self.nas_search_space[adapter_name])

        if naslora_mask is None:
            op_length = len(self.nas_search_space_dict[adapter_name])
            if layernorm:
                norm = LlamaRMSNorm(self.out_features, norm_eps)
                self.naslora_layernorms.update(
                    nn.ModuleDict(
                        {
                            adapter_name: nn.ModuleDict(
                                {
                                    _: LlamaRMSNorm(self.out_features, norm_eps)
                                    for _ in self.nas_search_space_dict[adapter_name]
                                }
                            )
                        }
                    )
                )
            self.naslora_mask[adapter_name] = torch.ones(
                op_length,
                dtype=self.weight.dtype,
            )
            self.naslora_weights.update(
                nn.ParameterDict(
                    {
                        adapter_name: nn.Parameter(
                            1e-3
                            * torch.ones(
                                len(self.nas_search_space_dict[adapter_name]),
                                dtype=self.weight.dtype,
                            )
                        )
                    }
                )
            )

        else:
            assert len(naslora_mask) == len(self.nas_search_space_dict[adapter_name]), "please check naslora_mask"
            self.naslora_weights = None
            self.naslora_mask[adapter_name] = torch.tensor(naslora_mask)
            self.prune = True
        nas_module_list = nn.ModuleList()
        # harmonize dropout settings
        for key, module in self.nas_search_space_dict[adapter_name].items():
            peft_type = module.pop("peft_type", None)
            if peft_type and naslora_dropout > 0.0 and peft_type != "zero":
                # set dropout to 0.0 for each submodule
                module[PEFT_DROPOUT_MAPPING[peft_type]] = 0.0
            activate_fn = module.get("activate_fn", None)
            peft_activated_fn = "" if not activate_fn else activate_fn
            self.activate_fn[adapter_name].append(peft_activated_fn)
        # Actual trainable parameters
        module_counter = 0
        for key, module in self.nas_search_space_dict[adapter_name].items():
            if self.naslora_mask[adapter_name][module_counter] == 1:
                nas_module_list.append(self._get_sub_module(key, module, init_naslora_weights))
            else:
                nas_module_list.append(nn.Identity())
            module_counter += 1

        if naslora_dropout > 0.0:
            naslora_dropout_layer = nn.Dropout(p=naslora_dropout)
        else:
            naslora_dropout_layer = nn.Identity()
        self.naslora_dropout.update(nn.ModuleDict({adapter_name: naslora_dropout_layer}))

        self.naslora_module.update(nn.ModuleDict({adapter_name: nas_module_list}))

        self.to(self.weight.device)


class Linear(nn.Linear, NASLoraLayer):
    # NASLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        nas_search_space: str,
        naslora_mask: list,
        reset_mode: int,
        naslora_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        layernorm: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_naslora_weights = kwargs.pop("init_naslora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        NASLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        # for additive lora adapters
        self.prune = False
        self.reset_mode = reset_mode
        self.layernorm = layernorm

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name,
            nas_search_space,
            naslora_dropout,
            init_naslora_weights,
            naslora_mask,
            norm_eps,
            layernorm,
        )
        self.active_adapter = adapter_name

    def save_top_n_adapters(self, top_N, naslora_dropout):
        if not self.prune:
            if self.layernorm:
                self.naslora_layernorms.pop(self.active_adapter)
            naslora_weights = self.naslora_weights.pop(self.active_adapter)
            self.naslora_weights_backup[self.active_adapter] = (
                naslora_weights.detach().cpu().to(torch.float16).numpy().tolist()
            )
            topk_index = torch.topk(naslora_weights, top_N).indices
            naslora_mask_new = torch.zeros(len(self.naslora_mask[self.active_adapter]))
            naslora_mask_new[topk_index] = 1
            self.naslora_mask[self.active_adapter] = naslora_mask_new

            just_b = False
            if self.reset_mode == 1:
                just_b = True
            for index, module in enumerate(self.naslora_module[self.active_adapter]):
                if self.reset_mode > 0:
                    if index in topk_index:
                        self.lora.reset_lora_parameters(
                            list(self.nas_search_space_dict[self.active_adapter].keys())[index],
                            just_b=just_b,
                        )
                for module_item in module:
                    if index in topk_index:
                        module_item.weight.requires_grad = True
                    else:
                        module_item.weight.requires_grad = False

            self.prune = True

    def merge(self):
        # TODO: for naslora
        if self.active_adapter not in self.naslora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.nas_search_space[self.active_adapter] != "" > 0:
            self.weight.data += (
                transpose(
                    self.naslora_B[self.active_adapter].weight @ self.naslora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        # TODO: for lora
        if self.active_adapter not in self.naslora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.nas_search_space[self.active_adapter] != "" > 0:
            self.weight.data -= (
                transpose(
                    self.naslora_B[self.active_adapter].weight @ self.naslora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if getattr(self.naslora_module, self.active_adapter, None) is None:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.nas_search_space[self.active_adapter] != "" and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.nas_search_space[self.active_adapter] != "" and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            additive_adapters_results = []
            for mask, key, module in zip(
                self.naslora_mask[self.active_adapter],
                self.nas_search_space_dict[self.active_adapter].keys(),
                self.naslora_module[self.active_adapter],
            ):
                if mask != 0:
                    if type(module) != nn.Identity:
                        activate_fn = ACTIVATE_FN_MAPPING.get(self.lora.activate_fn[key], nn.Identity())

                        if self.lora.d_parts[key] > 1:
                            result_temp = LoRA.Linear.d_parts_calcaulate(
                                x,
                                module,
                                self.lora.split_position[key],
                                self.lora.part_each_indexs[key],
                                self.lora.scaling[key],
                                activate_fn,
                                self.naslora_dropout[self.active_adapter],
                            )

                        else:
                            result_temp = LoRA.Linear.lora_calculate(
                                x,
                                self.lora.lora_A[key],
                                self.lora.lora_B[key],
                                self.lora.scaling[key],
                                activate_fn,
                                self.naslora_dropout[self.active_adapter],
                            )

                        if self.prune == False and self.layernorm:
                            # make the layer normalization work
                            result_temp = self.naslora_layernorms[self.active_adapter][key](result_temp)
                        additive_adapters_results.append(result_temp)
            if len(additive_adapters_results) > 0:
                additive_adapters_results_ = torch.stack(additive_adapters_results, 0)

                if self.prune == False:
                    naslora_weights = self.naslora_weights[self.active_adapter]
                    ones_index_matrix = torch.squeeze(torch.nonzero(naslora_weights), 1)
                    adapters_result = torch.einsum(
                        "i...,i->i...",
                        additive_adapters_results_,
                        nn.functional.softmax(naslora_weights)[ones_index_matrix],
                    )
                else:
                    adapters_result = additive_adapters_results_

                result += torch.sum(adapters_result, dim=0)

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result


if is_bnb_available():
    # TODO: for split and activate_fn
    class Linear8bitLt(bnb.nn.Linear8bitLt, NASLoraLayer):
        # NASLora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            nas_search_space: str,
            naslora_mask: list,
            naslora_dropout: float = 0.0,
            norm_eps: float = 1e-5,
            layernorm: bool = False,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            NASLoraLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_naslora_weights = kwargs.pop("init_naslora_weights", True)
            self.update_layer(
                adapter_name,
                nas_search_space,
                naslora_dropout,
                init_naslora_weights,
                naslora_mask,
                norm_eps,
                layernorm,
            )
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.naslora_A.keys():
                return result
            elif self.nas_search_space[self.active_adapter] != "" > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.naslora_B[self.active_adapter](
                            self.naslora_A[self.active_adapter](self.naslora_dropout[self.active_adapter](x))
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.naslora_B[self.active_adapter](
                            self.naslora_A[self.active_adapter](self.naslora_dropout[self.active_adapter](x))
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

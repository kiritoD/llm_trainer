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

from __future__ import annotations

from functools import partial
from logging import config
import math
from turtle import forward
import warnings
import re
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from altair import layer
import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
)
from ...utils import (
    TRANSFORMERS_MODELS_TO_HSSA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import HSSAConfig
from .layer import Linear, HSSALayer

NUMBER_RE = re.compile(r"\d+")


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


class HSSAModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (HSSA) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`HSSAConfig`]): The configuration of the HSSA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The HSSA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import HSSAConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = HSSAConfig(r=128)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`HSSAConfig`]): The configuration of the HSSA model.
    """

    name: str = "hssa"
    prefix: str = "hssa_lambda"
    prefixs: List[str] = ["hssa_lambda", "hssa_", "modules_weights_"]
    # if this argument is False, try updating each submodule for hssa_A, hssa_B and modules_weights, and then this flag will be set to True
    if_hierarchy_update: bool = False

    def __init__(self, model, config, adapter_name) -> None:
        self.reg = None
        super().__init__(model, config, adapter_name)

    def _prefix_in_key(self, key):
        for prefix in self.prefixs:
            if prefix in key:
                return True
        return False

    def reg_match(self, key: str) -> str | None:
        """find the target module name from the key

        Parameters
        ----------
        key : str
            key name, for example, "bert.encoder.layer.11.attention.self.query.weight"

        Returns
        -------
        str | None
            for example, "query"
        """
        if not self.reg:
            self.reg = re.compile(
                r"|".join(self.peft_config[self.active_adapter[0]].target_modules), re.I
            )
        results = self.reg.findall(key)
        if len(results) > 0:
            return results[0]
        return None

    def _hierarchical_update(self):

        if self.if_hierarchy_update:
            return

        # try allocating some part of the hssa_A, hssa_B, modules_weights_A and modules_weights_B matrix to the current target module
        # for reference, we should use dict not nn.ParameterDict to store the hssa_A, hssa_B and Modules_weights
        keys = [key for key, _ in self.model.named_modules() if self.name in key]
        for key in keys:
            parent, *_ = _get_submodules(self.model, key)
            if isinstance(parent, Linear) and not parent.if_hssa_update():
                target_name = self.reg_match(key)
                if not target_name:
                    continue
                else:
                    # try to allocate some part of the hssa_A, hssa_B, modules_weights_A and modules_weights_B matrix to the current target module
                    active_adapter = self.active_adapter[0]
                    hssa_config: HSSAConfig = self.peft_config[active_adapter]
                    hssa_A, hssa_B, modules_weights_A, modules_weights_B = (
                        {},
                        {},
                        {},
                        {},
                    )
                    if hssa_config.hierarchy_independent:
                        # before update the linear, we need to extract the corresponding part of the hssa_A and hssa_B
                        start, end = self.target_module_indexs[target_name]
                        if hssa_config.hierarchy_space_r != -1:
                            # 11
                            start *= hssa_config.hierarchy_space_r
                            end *= hssa_config.hierarchy_space_r
                        else:
                            # 10
                            start *= hssa_config.r
                            end *= hssa_config.r

                        hssa_A[active_adapter] = self.hssa_A[active_adapter][
                            start:end, :
                        ]

                        hssa_B[active_adapter] = self.hssa_B[active_adapter][
                            :, start:end
                        ]
                    else:
                        # 01 or 00
                        hssa_A = self.hssa_A
                        hssa_B = self.hssa_B

                    if hssa_config.hierarchy_space_r != -1:

                        start, end = self.target_module_indexs[target_name]
                        if hssa_config.hierarchy_independent:
                            # 11
                            start *= hssa_config.hierarchy_space_r
                            end *= hssa_config.hierarchy_space_r
                            modules_weights_A[active_adapter] = self.modules_weights_A[
                                active_adapter
                            ][:, start:end]
                            modules_weights_B[active_adapter] = self.modules_weights_B[
                                active_adapter
                            ][start:end, :]
                        else:
                            # 10
                            start *= hssa_config.r
                            end *= hssa_config.r
                            modules_weights_A[active_adapter] = self.modules_weights_A[
                                active_adapter
                            ][start:end, :]
                            modules_weights_B[active_adapter] = self.modules_weights_B[
                                active_adapter
                            ][:, start:end]
                    else:
                        modules_weights_A[active_adapter] = None
                        modules_weights_B[active_adapter] = None
                    layer_adaptive_param = None
                    if hssa_config.layer_adaptive:
                        layer_number = -1
                        try:
                            layer_number = int(NUMBER_RE.findall(key)[0])
                        except:
                            ...
                        if layer_number != -1:
                            layer_adaptive_param = self.layer_parameters_adaptive[
                                layer_number
                            ]

                    hierarchy_adaptive_A, hierarchy_adaptive_B = None, None
                    if hssa_config.hierarchy_adaptive:
                        start, end = self.target_module_indexs[target_name]
                        start *= hssa_config.hierarchy_space_r
                        end *= hssa_config.hierarchy_space_r
                        hierarchy_adaptive_A = self.subspace_parameters_adaptive_A[
                            start:end
                        ]
                        hierarchy_adaptive_B = self.subspace_parameters_adaptive_B[
                            start:end
                        ]

                    parent.update_hssa_space(
                        hssa_A,
                        hssa_B,
                        modules_weights_A,
                        modules_weights_B,
                        layer_adaptive_param,
                        hierarchy_adaptive_A,
                        hierarchy_adaptive_B,
                    )
        self.if_hierarchy_update = True
        # update the target module

    def _adaptive_init(self):
        """
        hierarchical space adaptive initialization.
        """
        active_adapter = self.active_adapter
        hssa_config: HSSAConfig = self.peft_config[active_adapter]
        layer_adaptive = hssa_config.layer_adaptive
        hierarchy_adaptive = hssa_config.hierarchy_adaptive
        hierarchy_r = hssa_config.hierarchy_space_r
        adaptive_threshold = hssa_config.adaptive_threshold
        self.num_hidden_layer = None
        self.adaptive_threshold = -1
        assert not ((adaptive_threshold != -1) ^ hierarchy_adaptive) or not (
            (adaptive_threshold != -1) ^ layer_adaptive
        ), "the argument `adaptive_threshold` in config should be set when `layer_adaptive` or `hierarchy_adaptive` are set to specific values."
        if layer_adaptive:
            self.num_hidden_layer = self.model.config.num_hidden_layers
            self.layer_parameters_adaptive = nn.Parameter(
                torch.ones(self.num_hidden_layer)
            )
        if hierarchy_adaptive:
            assert (
                hierarchy_r != -1
            ), "the argument `hierarchy_adaptive` of config should not be set to -1"

            r_multiple = len(hssa_config.target_modules)
            self.subspace_parameters_adaptive_A = nn.Parameter(
                torch.ones(hierarchy_r * r_multiple)
            )
            self.subspace_parameters_adaptive_B = nn.Parameter(
                torch.ones(hierarchy_r * r_multiple)
            )

        if adaptive_threshold != -1:
            self.adaptive_threshold = adaptive_threshold

    def forward(self, *args: warnings.Any, **kwargs: warnings.Any):

        # allocate some part of the hssa_A, hssa_B and Modules_weights matrix to the target modules
        self._hierarchical_update()

        return super().forward(*args, **kwargs)

    def merge_adapter(self, adapter_names: List[str] | None = None) -> None:
        # before merge, we need to allocate weihgts to the target modules
        self._hierarchical_update()
        return super().merge_adapter(adapter_names)

    def _find_dim(self, config) -> tuple[int, int]:
        """
        Finds the largest input and output dimensions across linear layers that have been wrapped with HSSA.

        This will be used for determining the size of the shared hssa_A and hssa_B matrices.
        """
        model_config = getattr(self.model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        largest_shape = None
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, (nn.Linear, Conv1D)):
                module_shape = tuple(module.weight.shape)
                if isinstance(module, Conv1D):
                    module_shape = module_shape[::-1]
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
                continue

            if module_shape != largest_shape:
                largest_shape = tuple(
                    max(a, b) for a, b in zip(largest_shape, module_shape)
                )

        if largest_shape is None:
            msg = "No layers types compatible with HSSA were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        return largest_shape

    def _prepare_model_for_hierarchical_space(self, config: HSSAConfig):
        """
        A helper method to prepare the model for hierarchical space.

        Parameters
        ----------
        config : HSSAConfig
            _description_
        """
        target_moduels: List[str] = config.target_modules
        target_moduels = sorted(target_moduels)
        self.r_multiple = len(target_moduels)
        self.target_module_indexs = {}
        for i, target_module in enumerate(target_moduels):
            self.target_module_indexs[target_module] = [
                i,
                (i + 1),
            ]

    def _init_hssa_A_hssa_B(self, config: HSSAConfig, adapter_name: str) -> None:
        linear_out_dim, linear_in_dim = self._find_dim(config)
        # for the adapter for different target modules, we need to use different hssa_A and hssa_B
        self._prepare_model_for_hierarchical_space(config)
        if config.tied_requires_grad:
            # if need grad, init hssa_A and hssa_B as parameters
            self.hssa_A = nn.ParameterDict({})
            self.hssa_B = nn.ParameterDict({})
        else:
            # use of persistent to exclude hssa_A and hssa_B from the state dict if we choose not to save them.
            self.hssa_A = BufferDict({}, persistent=config.save_projection)
            self.hssa_B = BufferDict({}, persistent=config.save_projection)

        self.modules_weights_A = nn.ParameterDict({})
        self.modules_weights_B = nn.ParameterDict({})
        # deterministic init of hssa_A and hssa_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(
            config.projection_prng_key
        )
        # patical function for efficency
        init_method = partial(_kaiming_init, generator=generator)
        """
        independent        | 0 or 1
        hierarchy_space_r  | 0 or 1 (0 mean -1, 1 mean a value which is > 0)
        00: hssa_A and hssa_B are shared across all target modules, modules_weights_A and modules_weights_B are None
        01: hssa_A and hssa_B are shared across all target modules, modules_weights_A and modules_weights_B are shared across the same type of modules
        10: hssa_A and hssa_B are independent for each target module, modules_weights_A and modules_weights_B are None
        11: hssa_A and hssa_B are independent for each target module, modules_weights_A and modules_weights_B are shared across the same type of modules
        """
        if config.hierarchy_independent:
            if config.hierarchy_space_r != -1:
                # 11
                hssa_A = init_method(
                    (config.hierarchy_space_r * self.r_multiple, linear_in_dim)
                )

                hssa_B = init_method(
                    (linear_out_dim, config.hierarchy_space_r * self.r_multiple)
                )
            else:
                # 10
                hssa_A = init_method((config.r * self.r_multiple, linear_in_dim))
                hssa_B = init_method((linear_out_dim, config.r * self.r_multiple))

        else:
            if config.hierarchy_space_r != -1:
                # 01
                hssa_A = init_method((config.hierarchy_space_r, linear_in_dim))
                hssa_B = init_method((linear_out_dim, config.hierarchy_space_r))
            else:
                # 00
                hssa_A = init_method((config.r, linear_in_dim))
                hssa_B = init_method((linear_out_dim, config.r))

        if config.hierarchy_space_r != -1:
            if config.hierarchy_independent:
                # 11
                modules_weights_A = init_method(
                    (config.r, config.hierarchy_space_r * self.r_multiple)
                )
                modules_weights_B = init_method(
                    (config.hierarchy_space_r * self.r_multiple, config.r)
                )
            else:
                # 01
                modules_weights_A = init_method(
                    (config.r * self.r_multiple, config.hierarchy_space_r)
                )
                modules_weights_B = init_method(
                    (config.hierarchy_space_r, config.r * self.r_multiple)
                )
        else:
            # 10 or 00
            modules_weights_A = None
            modules_weights_B = None

        # the function `_mark_only_adapters_as_trainable` only take effect in the target model not in self, so the following four parts are not affected.
        # this design is effective to decouple the behavior between the model and the layers
        self.modules_weights_A[adapter_name] = modules_weights_A
        self.modules_weights_B[adapter_name] = modules_weights_B

        self.hssa_A[adapter_name] = hssa_A
        self.hssa_B[adapter_name] = hssa_B

    def _pre_injection_hook(
        self, model: nn.Module, config: HSSAConfig, adapter_name: str
    ) -> None:
        # for adaptive initialization, we need to use less parameters
        self._adaptive_init()
        self._init_hssa_A_hssa_B(config, adapter_name)

    def _check_new_adapter_config(self, config: HSSAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        for existing_config in self.peft_config.values():
            if existing_config is config:
                # skip the current config
                continue

            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"HSSA PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = sorted(
            {config.save_projection for config in self.peft_config.values()}
        )
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "HSSA projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_project_unique_values}"
            )

    @staticmethod
    def _check_target_module_exists(hssa_config, key):
        return check_target_module_exists(hssa_config, key)

    def _create_and_replace(
        self,
        hssa_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = hssa_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "hssa_dropout": hssa_config.hssa_dropout,
            "fan_in_fan_out": hssa_config.fan_in_fan_out,
            "init_weights": hssa_config.init_weights,
        }
        kwargs["bias"] = bias
        # TODO: add quantization support

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                r,
                hssa_config.hssa_dropout,
                hssa_config.init_weights,
                d_initial=hssa_config.d_initial,
            )
        else:
            new_module = self._create_new_module(
                hssa_config,
                adapter_name,
                target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "hssa_" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "hssa_only":
                for m in model.modules():
                    if (
                        isinstance(m, HSSALayer)
                        and hasattr(m, "bias")
                        and m.bias is not None
                    ):
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(
                    f"Requested bias: {bias}, is not implemented."
                )

    @staticmethod
    def _create_new_module(hssa_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = hssa_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = hssa_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            target,
            adapter_name,
            bias=bias,
            d_initial=hssa_config.d_initial,
            **kwargs,
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in asdict(value).items()
            }
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, HSSALayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_HSSA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_HSSA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        # we cannot use self.prefix as we want to include non-trainable hssa parameters
        key_list = [key for key, _ in self.model.named_modules() if "hssa" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)

                self._replace_module(
                    parent, target_name, target.get_base_layer(), target
                )
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(
                    parent, target_name, target.modules_to_save[target.active_adapter]
                )

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable hssa parameters
        key_list = [key for key, _ in self.model.named_modules() if "hssa" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, HSSALayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        r"""
        This method merges the HSSA layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Gets back the base model by removing all the HSSA modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

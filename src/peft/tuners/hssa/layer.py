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

from sys import modules
import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class HSSALayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights, here, we don't move the modules weights, because these weights will move to device along with the base model
    adapter_layer_names = ("hssa_lambda_b", "hssa_lambda_d")
    # this follow the base model, so remove
    # other_param_names = ("hssa_A", "hssa_B")
    other_param_names = ()

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.hssa_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.hssa_lambda_b = nn.ParameterDict({})
        self.hssa_lambda_d = nn.ParameterDict({})

        # Stores a reference to the hssa_A/B BufferDict or nn.ParameterDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.hssa_A: Optional[Union[BufferDict, nn.ParameterDict]] = None
        self.hssa_B: Optional[Union[BufferDict, nn.ParameterDict]] = None
        self.modules_weights_A: Optional[nn.ParameterDict] = None
        self.modules_weights_B: Optional[nn.ParameterDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape
                if hasattr(base_layer.weight, "ds_shape")
                else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def if_hssa_update(self):
        return (
            self.hssa_A
            and self.hssa_B
            and self.modules_weights_A
            and self.modules_weights_B
        )

    def update_hssa_space(
        self,
        hssa_A: Union[BufferDict, nn.ParameterDict],
        hssa_B: Union[BufferDict, nn.ParameterDict],
        modules_weights_A: Optional[nn.ParameterDict] = None,
        modules_weights_B: Optional[nn.ParameterDict] = None,
    ):
        self.hssa_A = hssa_A
        self.hssa_B = hssa_B
        self.modules_weights_A = modules_weights_A
        self.modules_weights_B = modules_weights_B

    def update_layer(
        self,
        adapter_name,
        r,
        hssa_dropout,
        init_weights,
        d_initial: float = 0.1,
    ):
        if r <= 0:
            raise ValueError(
                f"`r` should be a positive integer value but the value passed is {r}"
            )
        self.r[adapter_name] = r
        if hssa_dropout > 0.0:
            hssa_dropout_layer = nn.Dropout(p=hssa_dropout)
        else:
            hssa_dropout_layer = nn.Identity()

        self.hssa_dropout.update(nn.ModuleDict({adapter_name: hssa_dropout_layer}))
        # Actual trainable parameters
        self.hssa_lambda_b[adapter_name] = nn.Parameter(
            torch.ones(self.out_features), requires_grad=True
        )
        self.hssa_lambda_d[adapter_name] = nn.Parameter(
            torch.randn(r), requires_grad=True
        )

        if init_weights:
            self.reset_hssa_parameters(adapter_name, d_initial=d_initial)

        # here the modules weights won't be move to device, because they will be moved to device along with the base model, if you want, should add the name to constants of this layer
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_hssa_parameters(self, adapter_name, d_initial: float = 0.1):
        if adapter_name in self.hssa_lambda_d.keys():
            with torch.no_grad():
                nn.init.zeros_(self.hssa_lambda_d[adapter_name]).fill_(d_initial)
                nn.init.zeros_(self.hssa_lambda_b[adapter_name])


class Linear(nn.Linear, HSSALayer):
    # HSSA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        hssa_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        HSSALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.update_layer(
            adapter_name,
            r,
            hssa_dropout,
            init_weights,
            d_initial=d_initial,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None
    ) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        # todo (djw has done): add support for merging multiple adapters for hssa
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hssa_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        # todo (djw has done): add support for unmerging multiple adapters for hssa
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.hssa_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(
                    active_adapter
                )

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # todo (djw has done): add support for getting delta weight for hssa.
        hssa_A = self.hssa_A[adapter]
        hssa_B = self.hssa_B[adapter]
        modules_wegihts_A = None
        modules_wegihts_B = None
        if self.modules_weights_A is not None:
            modules_wegihts_A = self.modules_weights_A[adapter]
            modules_wegihts_B = self.modules_weights_B[adapter]

        device = hssa_B.device
        dtype = hssa_B.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lambda_d = self.hssa_lambda_d[adapter]
        lambda_b = self.hssa_lambda_b[adapter]

        if cast_to_fp32:
            hssa_A = hssa_A.float()
            hssa_B = hssa_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()
            if modules_wegihts_A is not None:
                modules_wegihts_A = modules_wegihts_A.float()
                modules_wegihts_B = modules_wegihts_B.float()

        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        if modules_wegihts_A is not None:
            # based on the inference, we can deduce from the formula.
            # output = x @ W_A.T @ m_A.T * lambda_d @ m_B.T @ W_B.T * lambda_b
            output_tensor = transpose(
                (lambda_b * hssa_B @ modules_wegihts_B)
                @ (lambda_d * modules_wegihts_A @ hssa_A),
                self.fan_in_fan_out,
            )
        else:
            output_tensor = transpose(
                (lambda_b * hssa_B) @ (lambda_d * hssa_A), self.fan_in_fan_out
            )

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            # TODO: why?
            self.hssa_lambda_d[adapter].data = lambda_d.to(dtype)
            self.hssa_lambda_b[adapter].data = lambda_b.to(dtype)

        return output_tensor

    def hssa_caculate(
        self,
        x,
        lambda_d,
        lambda_b,
        hssa_A,
        module_weight_A,
        module_weight_B,
        hssa_B,
        dropout,
    ):
        # As adapted layers may have different shapes and HSSA contains a single shared pair of A and B matrices,
        # we initialize these matrices with the largest required size for each dimension.
        # During the forward pass, required submatrices are sliced out from the shared hssa_A and hssa_B.
        if module_weight_A is not None:
            result_1_after_hssa_A = F.linear(dropout(x), hssa_A)
            result_2_after_module_A = lambda_d * F.linear(
                result_1_after_hssa_A, module_weight_A
            )
            result_3_after_module_B = F.linear(result_2_after_module_A, module_weight_B)
            result_4_afater_hssa_B = lambda_b * F.linear(
                result_3_after_module_B, hssa_B
            )
            result = result_4_afater_hssa_B
        else:
            sliced_A = hssa_A
            sliced_B = hssa_B
            result = lambda_b * F.linear(
                lambda_d * F.linear(dropout(x), sliced_A), sliced_B
            )

        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hssa_lambda_d.keys():
                    continue

                lambda_d = self.hssa_lambda_d[active_adapter]
                lambda_b = self.hssa_lambda_b[active_adapter]

                hssa_A = self.hssa_A[active_adapter]
                module_weight_A = self.modules_weights_A[active_adapter]
                module_weight_B = self.modules_weights_B[active_adapter]
                hssa_B = self.hssa_B[active_adapter]

                dropout = self.hssa_dropout[active_adapter]
                x = x.to(lambda_d.dtype)
                # the caculate function is implemented in self.hssa_caculate, which is called for each active adapter
                result = result + self.hssa_caculate(
                    x,
                    lambda_d,
                    lambda_b,
                    hssa_A,
                    module_weight_A,
                    module_weight_B,
                    hssa_B,
                    dropout,
                )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hssa." + rep

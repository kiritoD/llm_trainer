import torch
import torch.nn as nn
import torch.nn.functional as F
from ....utils.auxiliary import load_yaml

from ..lora_a import layer
from ..lora_a import ACTIVATE_FN_MAPPING, Lora_ALayer

from ...import_utils import is_bnb_available

from ...utils import transpose

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
            self.lora = Lora_ALayer(self.in_features, self.out_features)
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
                            result_temp = layer.Linear.d_parts_calcaulate(
                                x,
                                module,
                                self.lora.split_position[key],
                                self.lora.part_each_indexs[key],
                                self.lora.scaling[key],
                                activate_fn,
                                self.naslora_dropout[self.active_adapter],
                            )

                        else:
                            result_temp = layer.Linear.lora_calculate(
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
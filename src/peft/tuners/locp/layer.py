import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import (
    transpose,
)
from ...import_utils import is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

ACTIVATE_FN_MAPPING = dict(
    relu=nn.ReLU(), sigmoid=nn.Sigmoid(), gelu=nn.GELU(), silu=nn.SiLU()
)


class LoCPLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # extra setting
        self.lora_AB = nn.ModuleDict({})
        self.lora_cp_x = nn.Parameter(torch.ones(in_features).unsqueeze(0))
        self.lora_cp_y = nn.Parameter(torch.ones(out_features).unsqueeze(1))
        self.cp_x = nn.Parameter(torch.ones(in_features).unsqueeze(0))
        self.cp_y = nn.Parameter(torch.ones(out_features).unsqueeze(1))
        self.d_parts = {}
        self.part_each_indexs = {}
        self.split_position = {}
        self.activate_fn = {}
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        d_parts,
        activate_fn,
        to_device=True,
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.d_parts[adapter_name] = int(d_parts)
        self.activate_fn[adapter_name] = activate_fn
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        # self.cp_x.update(
        #     nn.ModuleDict({adapter_name: nn.Parameter(torch.ones(self.in_features))})
        # )
        # self.cp_y.update(
        #     nn.ModuleDict({adapter_name: nn.Parameter(torch.ones(self.out_features))})
        # )

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            if self.d_parts[adapter_name] == 1:
                self.lora_A.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Linear(self.in_features, r, bias=False)}
                    )
                )
                self.lora_B.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                    )
                )
            elif self.d_parts[adapter_name] > 1:
                self.split_position[adapter_name] = None
                self.part_each_indexs[adapter_name] = [0]
                split_number = 1
                if self.in_features <= self.out_features:
                    self.split_position[adapter_name] = "out"
                    split_number = self.out_features // self.d_parts[adapter_name]
                else:
                    self.split_position[adapter_name] = "in"
                    split_number = self.in_features // self.d_parts[adapter_name]

                module_list = []
                for _ in range(self.d_parts[adapter_name]):
                    split_number_temp = split_number
                    if _ == self.d_parts[adapter_name] // 2:
                        split_number_temp += (
                            self.out_features % self.d_parts[adapter_name]
                        )
                    self.part_each_indexs[adapter_name].append(
                        self.part_each_indexs[adapter_name][-1] + split_number_temp
                    )
                    module_list.append(
                        nn.Linear(
                            (
                                split_number_temp
                                if self.split_position[adapter_name] == "in"
                                else self.in_features
                            ),
                            r,
                            bias=False,
                        )
                    )
                    module_list.append(
                        nn.Linear(
                            r,
                            (
                                split_number_temp
                                if self.split_position[adapter_name] == "out"
                                else self.out_features
                            ),
                            bias=False,
                        )
                    )
                self.part_each_indexs[adapter_name].append(
                    self.out_features
                    if self.split_position[adapter_name] == "out"
                    else self.in_features
                )
                self.lora_AB.update(
                    nn.ModuleDict(
                        {
                            adapter_name: nn.ModuleList(module_list),
                        }
                    )
                )
            else:
                raise ValueError(f"`d_parts` shoud >= 1")
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        if to_device:
            self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name, just_b=False):
        if self.d_parts[adapter_name] == 1:
            if adapter_name in self.lora_A.keys():
                # initialize A the same way as the default for nn.Linear and B to zero
                if not just_b:
                    nn.init.kaiming_uniform_(
                        self.lora_A[adapter_name].weight, a=math.sqrt(5)
                    )
                nn.init.zeros_(self.lora_B[adapter_name].weight)
        else:
            if adapter_name in self.lora_AB.keys():
                for index in range(0, len(self.lora_AB[adapter_name]), 2):
                    if not just_b:
                        nn.init.kaiming_uniform_(
                            self.lora_AB[adapter_name][index].weight,
                            a=math.sqrt(5),
                        )
                    nn.init.zeros_(self.lora_AB[adapter_name][index + 1].weight)


class Linear(nn.Linear, LoCPLayer):
    # LoCP implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        d_parts: int = 1,
        activate_fn: str = None,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoCPLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            d_parts,
            activate_fn,
        )
        self.active_adapter = adapter_name

    def merge(self):
        # TODO: for split and activate_fn
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight
                    @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        # TODO: for split and activate_fn
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight
                    @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    @classmethod
    def d_parts_calcaulate(
        cls,
        x,
        module,
        split_position,
        part_each_indexs,
        scaling,
        activate_fn,
        dropout,
    ):
        result = 0
        results_split = []
        for index in range(0, len(module), 2):
            if split_position == "in":
                results_split.append(
                    module[index + 1](
                        activate_fn(
                            module[index](
                                dropout(
                                    x[
                                        :,
                                        :,
                                        part_each_indexs[index // 2] : part_each_indexs[
                                            index // 2 + 1
                                        ],
                                    ]
                                )
                            )
                        )
                    )
                )
            else:
                results_split.append(
                    module[index + 1](activate_fn(module[index](dropout(x))))
                )
        if split_position == "out":
            result += torch.cat(results_split, -1) * scaling
        else:
            for _ in results_split:
                result += _ * scaling
        return result

    @classmethod
    def lora_calculate(cls, x, module_A, module_B, scaling, activate_fn, dropout):
        x = x.to(module_A.weight.dtype)
        return module_B(activate_fn(module_A(dropout(x)))) * scaling

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if (
            getattr(self.lora_A, self.active_adapter, None) is None
            and getattr(self.lora_AB, self.active_adapter, None) is None
        ):
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            # all
            result = F.linear(
                x,
                transpose(
                    self.weight * self.lora_cp_x * self.lora_cp_y, self.fan_in_fan_out
                ),
                bias=self.bias,
            )
            # just x
            # result = F.linear(
            #     x,
            #     transpose(self.weight * self.lora_cp_x, self.fan_in_fan_out),
            #     bias=self.bias,
            # )
            # just y
            # result = F.linear(
            #     x,
            #     transpose(self.weight * self.lora_cp_y, self.fan_in_fan_out),
            #     bias=self.bias,
            # )

            # activate_fn = ACTIVATE_FN_MAPPING.get(
            #     self.activate_fn[self.active_adapter], nn.Identity()
            # )
            # if self.d_parts[self.active_adapter] == 1:
            #     result += self.lora_calculate(
            #         x,
            #         self.lora_A[self.active_adapter],
            #         self.lora_B[self.active_adapter],
            #         self.scaling[self.active_adapter],
            #         activate_fn,
            #         self.lora_dropout[self.active_adapter],
            #     )
            # else:
            #     result += self.d_parts_calcaulate(
            #         x,
            #         self.lora_AB[self.active_adapter],
            #         self.split_position[self.active_adapter],
            #         self.part_each_indexs[self.active_adapter],
            #         self.scaling[self.active_adapter],
            #         activate_fn,
            #         self.lora_dropout[self.active_adapter],
            #     )
        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)

        return result


if is_bnb_available():
    # TODO: for split and activate_fn
    class Linear8bitLt(bnb.nn.Linear8bitLt, LoCPLayer):
        # LoCP implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get(
                    "memory_efficient_backward", False
                ),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoCPLayer.__init__(self, in_features=in_features, out_features=out_features)

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(
                adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
            )
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](
                                self.lora_dropout[self.active_adapter](x)
                            )
                        ).to(expected_dtype)
                        * self.scaling[self.active_adapter]
                    )
                else:
                    output = (
                        self.lora_B[self.active_adapter](
                            self.lora_A[self.active_adapter](
                                self.lora_dropout[self.active_adapter](x)
                            )
                        )
                        * self.scaling[self.active_adapter]
                    )
                result += output
            return result

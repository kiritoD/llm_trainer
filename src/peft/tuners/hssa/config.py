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

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ...config import PeftConfig
from ...utils import PeftType


@dataclass
class HSSAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`HSSAModel`].

    Paper: wip.

    Args:
        r (`int`, *optional*, defaults to `256`):
            HSSA parameter dimension ("rank"). Choose higher values than LoRA ranks here, since HSSA uses far fewer
            parameters than LoRA (see Table 1).
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply HSSA to. Only linear layers are supported.
        projection_prng_key (`int`):
            HSSA PRNG init key. Used for initialising hssa_A and hssa_B for new models or when loading a checkpoint
            that did not include these projections. Defaults to `0`.
        save_projection (`bool`):
            Whether to save the hssa_A / hssa_B projections in the state dict alongside per layer lambda_b / lambda_d
            weights. This will increase the size of the checkpoint, but guarantee that we can reload the checkpoint on
            all system configurations. Defaults to `True`.
        hssa_dropout (`float`):
            The dropout probability for HSSA layers.
        d_initial (`float`, *optional*, defaults to `0.1`):
            Initial init value for `hssa_lambda_d` vector used when initializing the HSSA parameters. Small values
            (<=0.1) are recommended (see Table 6c in the paper).
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for HSSA. Can be 'none', 'all' or 'hssa_only'. If 'all' or 'hssa_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from HSSA layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the HSSA layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the HSSA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the HSSA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    hierarchy_space_r: int = field(
        default=128, metadata={"help": "HSSA hierarchy space dimension"}
    )
    r: int = field(default=256, metadata={"help": "HSSA attention dimension"})
    tied_requires_grad: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to tie and update the hssa_A and hssa_B projection matrices. If True, the projection matrices will be "
                "shared and updated between the hssa_A and hssa_B matrices, which can help with stability and reduce the number of "
                "parameters."
            )
        },
    )
    hierarchy_independent: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use independent HSSA_A and HSSA_B projection matrices for different target modules."
            )
        },
    )
    layer_adaptive: bool = field(
        default=False,
        metadata={"help": "Whether to use layer-adaptive HSSA (L-HSSA) or not."},
    )
    hierarchy_adaptive: bool = field(
        default=False,
        metadata={"help": "Whether to use hierarchy-adaptive HSSA (H-HSSA) or not."},
    )
    adaptive_percent: float = field(
        default=0.8,
        metadata={
            "help": "The percentage of the total number of `hierarchy_space_r` to be employed for HSSA."
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with HSSA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    projection_prng_key: int = field(
        default=0,
        metadata={
            "help": (
                "HSSA PRNG init key. Used for initialising hssa_A and hssa_B for new models or when loading a "
                "checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the hssa_A / hssa_B projections in the state dict alongside per layer lambda_b / "
                "lambda_d weights. This will increase the size of the checkpoint, but guarantee that we can reload "
                "the checkpoint on all system configurations."
            )
        },
    )
    hssa_dropout: float = field(default=0.0, metadata={"help": "HSSA dropout"})
    d_initial: float = field(
        default=0.1, metadata={"help": "Initial init value for d vector."}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for HSSA. Can be 'none', 'all' or 'hssa_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from HSSA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the HSSA layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.HSSA
        self.target_modules = (
            set(self.target_modules)
            if isinstance(self.target_modules, list)
            else self.target_modules
        )

        if not self.save_projection:
            warnings.warn(
                "Specified to not save hssa_A and hssa_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )

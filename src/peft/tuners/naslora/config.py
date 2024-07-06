
from dataclasses import dataclass, field
from typing import List, Optional, Union
from ..lora import LoraConfig
from ...utils import PeftType

@dataclass
class NASLoraConfig(LoraConfig):
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


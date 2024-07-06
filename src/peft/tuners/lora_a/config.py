from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union
from ..lora.config import LoraConfig
from ...utils import PeftType


@dataclass
class Lora_AConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`Lora_a_Model`].

    Args:
        r (`int`): Lora_A attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora_A to.
        Lora_a__alpha (`float`): The alpha parameter for Lora_A scaling.
        Lora_a__dropout (`float`): The dropout probability for Lora_A layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora_A. Can be 'none', 'all' or 'Lora_a__only'
        modules_to_save (`List[str]`):List of modules apart from Lora_A layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora_A attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora_A."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    Lora_a__alpha: int = field(default=None, metadata={"help": "Lora_A alpha"})
    Lora_a__dropout: float = field(default=None, metadata={"help": "Lora_A dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora_A. Can be 'none', 'all' or 'Lora_a__only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Lora_A layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_Lora_a__weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora_A layers."},
    )
    # modify Lora_A
    d_parts: bool = field(
        default=1,
        metadata={"help": "to divide linear into `d_parts` parts"},
    )
    activate_fn: Optional[str] = field(
        default=None,
        metadata={
            "help": "whther add activated function between Lora_a__A and Lora_a__B modules"
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA_A


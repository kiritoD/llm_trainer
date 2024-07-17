from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AdaptiveBaseConfig:
    target_module: Optional[List[str]] = field(
        default=None, metadata={"help": "The target module to be fine-tuned"}
    )
    module_policy: Optional[str] = field(
        default="vector",
        metadata={"help": "The policy to select the module to be fine-tuned"},
    )
    layer_policy: Optional[str] = field(
        default="vector",
        metadata={"help": "The policy to select the layer to be fine-tuned"},
    )

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers.training_args import TrainingArguments


class CommonWriter:
    def __init__(
        self,
        params: Dict[str, Any],
        model: torch.nn.Module,
        infer_dataset: Dataset,
        data_collator: Optional[Callable] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ):
        self.params = params
        self.model = model
        self.dataset = infer_dataset
        self.collator = data_collator
        self.tokenizer = tokenizer

    def inference(self, test_dataset_loader, dist_params):
        device = torch.device("cuda", dist_params["current_gpu"])

from functools import partial

import torch
from transformers import AutoConfig

from ..utils.logging import get_logger, rank_zero_info

logger = get_logger("Model")
rank_zero_info = partial(rank_zero_info, logger=logger)

ModelMapping = {
    "Causal": "get_causal_lm",
    "SequenceClassification": "get_sequence_classification_model",
}

class Model:
    def __init__(self, params) -> None:
        self.params = params

    @classmethod
    def count_pramameter(cls, model, verbose=False):
        trainable_parameters = 0
        total_parameters = 0
        for p in model.parameters():
            p_number = p.numel()
            total_parameters += p_number
            if p.requires_grad:
                trainable_parameters += p_number
        if verbose == True:
            rank_zero_info(
                f"Total parameters: {total_parameters:,d} || Trainable parameters:{trainable_parameters:,d} || { (trainable_parameters/total_parameters):.2%}"
            )
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "trainabl_total%": trainable_parameters * 100 / total_parameters,
        }

    def get_causal_lm(self, load_weight_after_peft=False, **args):
        from transformers import AutoModelForCausalLM

        model_pretrained_path = self.params["pretrain_model_path"]

        self.config = AutoConfig.from_pretrained(model_pretrained_path, trust_remote_code=True)
        if not load_weight_after_peft:
            # if some checkpoint exists, will load this weight file
            model_pretrained_path_ = (
                model_pretrained_path
                if not self.params.get("checkpoint_model_path", None) or self.params["peft"]
                else self.params["checkpoint_model_path"]
            )
            torch_float = torch.float16
            torch_type = self.params.get("torch_dtype", None)
            if torch_type:
                torch_float = getattr(torch, self.params["torch_dtype"], torch.float16)
            else:
                torch_type = "auto"
            causal_model = AutoModelForCausalLM.from_pretrained(
                model_pretrained_path_,
                config=self.config,
                low_cpu_mem_usage=self.params.get("low_cpu_mem_usage", False),
                torch_dtype=torch_float,
                trust_remote_code=True,
            )
            # causal_model.config.use_cache = False
        else:
            causal_model = AutoModelForCausalLM.from_config(config=self.config)

        return causal_model
    
    def get_sequence_classification_model(self, num_labels: int = 0, finetuning_task: str = None, **args):
        from transformers import AutoModelForSequenceClassification

        model_pretrained_path = self.params["pretrain_model_path"]

        self.config = AutoConfig.from_pretrained(
            model_pretrained_path, 
            num_labels=num_labels,
            finetuning_task=finetuning_task,
            cache_dir=self.params.get("cache_dir", None),
            revision=self.params.get("model_revision", "main"),
            token=self.params.get("token", None),
            trust_remote_code=self.params.get("trust_remote_code", True),
        )
        
        # if some checkpoint exists, will load this weight file
        model_pretrained_path_ = (
            model_pretrained_path
            if not self.params.get("checkpoint_model_path", None) or self.params["peft"]
            else self.params["checkpoint_model_path"]
        )
        # torch_float = torch.float16
        # torch_type = self.params.get("torch_dtype", None)
        # if torch_type:
        #     torch_float = getattr(torch, self.params["torch_dtype"], torch.float16)
        # else:
        #     torch_type = "auto"
        model = AutoModelForSequenceClassification.from_pretrained(
                model_pretrained_path_,
                from_tf=bool(".ckpt" in model_pretrained_path_),
                config=self.config,
                low_cpu_mem_usage=self.params.get("low_cpu_mem_usage", False),
                # torch_dtype=torch_float,
                trust_remote_code=self.params.get("trust_remote_code", True),
                cache_dir=self.params.get("cache_dir", None),
                revision=self.params.get("model_revision", "main"),
                token=self.params.get("token", None),
                ignore_mismatched_sizes=self.params.get("ignore_mismatched_sizes", False),
        )

        return model
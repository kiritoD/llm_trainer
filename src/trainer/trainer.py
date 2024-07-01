import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import jsonlines
import numpy as np
import src.utils.metric as metric_fn
import torch
import torch.distributed as dist
from src.models.model import Model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GenerationConfig,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainingArguments,
    is_datasets_available,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.optimization import get_scheduler
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import PredictionOutput, seed_worker
from transformers.utils import is_sagemaker_mp_enabled

UNDERLINE = chr(9601)

from transformers.trainer import *
from transformers.training_args import TrainingArguments

from ..search.policy import Search_Policy
from ..utils.logging import get_logger

logger = get_logger("Trainer")

class StepLR(torch.optim.lr_scheduler.StepLR):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            self.optimizer.param_groups[0]["lr"] * self.gamma,
            self.optimizer.param_groups[1]["lr"],
        ]


class ConsineLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (self.optimizer.param_groups[0]["lr"] - self.eta_min)
            + self.eta_min,
            self.optimizer.param_groups[1]["lr"],
        ]


@dataclass
class CausalFtTrainingArguments(Seq2SeqTrainingArguments):
    wandb_project_name: str = field(
        default="LLM_TRAIN",
        metadata={"help": ("wandb project name wich contains a series of runs")},
    )
    peft: bool = field(
        default=True,
        metadata={"help": ("if train peft model")},
    )

    prediction_file_name: Optional[str] = field(
        default=None,
        metadata={"help": ("The `prediction_file_name` to be use for output results")},
    )
    care_tokens: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("the care tokens probs")},
    )
    eval_fn: Optional[str] = field(
        default="",
        metadata={"help": ("evaluation for dataset")},
    )
    eval_fn_params: Optional[dict] = field(
        default=None,
        metadata={"help": ("specification for dataset")},
    )

@dataclass
class SequenceClassifyTrainingArguments(TrainingArguments):
    wandb_project_name: str = field(
        default="LLM_TRAIN",
        metadata={"help": ("wandb project name wich contains a series of runs")},
    )
    peft: bool = field(
        default=True,
        metadata={"help": ("if train peft model")},
    )
    
    eval_fn: Optional[str] = field(
        default="",
        metadata={"help": ("evaluation for dataset")},
    )
    eval_fn_params: Optional[dict] = field(
        default=None,
        metadata={"help": ("specification for dataset")},
    )
    

class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    def save_top_n_adapters(self, model, peft_config, state):
        if (peft_config.search_strategy == "step" and state.global_step == peft_config.search_step) or (
            peft_config.search_strategy == "epoch" and state.epoch >= peft_config.search_step
        ):
            model.save_top_n_adapters()
            model.print_trainable_parameters()

    def output_log(self, args: CausalFtTrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                label_loss_ = data.get("label_loss", "")
                causal_loss_ = data.get("causal_loss", "")
                reg_loss_ = data.get("reg_loss", "")
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                first_arch_param_ = data.get("first_arch_param", "")
                last_arch_param_ = data.get("last_arch_param", "")

                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10} || label_loss: {label_loss_:<10} || causal_loss: {causal_loss_:<10} || reg_loss: {reg_loss_:<10} || first_arch_param: {first_arch_param_} || last_arch_param: {last_arch_param_}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        os.makedirs(args.output_dir, exist_ok=True)
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def evaluate_check(self, model, args: CausalFtTrainingArguments, state):
        peft_str = "peft"
        save_number = str(state.global_step)
        peft_type = ""
        if args.peft and peft_str in str(type(model)):
            peft_type = getattr(model.peft_type, "name", None)
            peft_type = "peft" if not peft_type else peft_type
        checkpoint_name = "step_" if args.peft else "checkpoint-"
        if self.args_check(args):
            args.prediction_file_name = os.path.join(
                args.output_dir,
                peft_type,
                checkpoint_name + save_number,
                args.prediction_file_name.split(os.path.sep)[-1],
            )

    def merge_files(self, prediction_file_name):
        re_prediction_file_name = prediction_file_name.replace(".jsonl", "_*")
        old_files = glob.glob(re_prediction_file_name)
        sql_set = set()
        with open(prediction_file_name, "w") as writer:
            for file_name in old_files:
                with jsonlines.Reader(open(file_name, "r")) as reader:
                    for line in reader:
                        line_str = json.dumps(line)
                        line_hash = hash(f"{line['question'] + line['data_id']}")
                        if line_hash not in sql_set:
                            writer.write(line_str + "\n")
                            sql_set.add(line_hash)
            writer.write(line_str + "\n")

        os.system(f"rm {re_prediction_file_name}")

    def peft_save(self, model, args, state):
        peft_str = "peft"
        model_ = model
        save_number = str(state.global_step)
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            if args.do_eval and self.args_check(args):
                self.merge_files(args.prediction_file_name)
            # for adapter save
            if args.peft and peft_str in str(type(model_)) and not is_deepspeed_zero3_enabled():
                # if model is peft-based, save the extra weights, zero3 not supprot
                peft_type = getattr(model_.peft_type, "name", None)
                peft_type = "peft" if not peft_type else peft_type
                epoch = "step_" + save_number
                output_dir = os.path.join(args.output_dir, peft_type, epoch)
                os.makedirs(output_dir, exist_ok=True)
                model_.save_pretrained(output_dir)
    def args_check(self, args):
        return "CausalFtTrainingArguments" in str(type(args))
    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model_ = kwargs["model"]
        peft_config = getattr(model_, "active_peft_config", None)
        # for naslora search operator
        if peft_config and peft_config.peft_type.value == "NASLORA":
            Search_Policy.policy_handler(state, model_, peft_config)
        return super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(
        self,
        args: CausalFtTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model_ = kwargs["model"]
        if state.global_step % 10 == 0:
            x = 1
        peft_config = getattr(model_, "active_peft_config", None)
        # if evalue via generate
        if args.do_eval:
            self.evaluate_check(model_, args, state)
        # for naslora search operator
        if peft_config and peft_config.peft_type.value == "NASLORA":
            self.save_top_n_adapters(model_, peft_config, state)
            if model_.prune == True:
                model_.save_adapter_weights(args.output_dir)
        return super().on_step_end(args, state, control, **kwargs)

    def on_epoch_end(
        self,
        args: CausalFtTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        self.output_log(args, state)
        model_ = kwargs["model"]
        # if evalue via generate
        if args.do_eval:
            self.evaluate_check(model_, args, state)
        else:
            self.peft_save(kwargs["model"], args, state)

        # for search operator
        peft_config = getattr(model_, "active_peft_config", None)
        if peft_config and peft_config.peft_type.value == "NASLORA":
            self.save_top_n_adapters(model_, peft_config, state)
            if model_.prune == True:
                model_.save_adapter_weights(args.output_dir)
        return super().on_epoch_end(args, state, control, **kwargs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.peft_save(kwargs["model"], args, state)
        return super().on_evaluate(args, state, control, **kwargs)

    def on_predict(
        self,
        args: CausalFtTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            self.merge_files(args.prediction_file_name)
            if getattr(args, "eval_fn", None):
                eval_fn = getattr(metric_fn, args.eval_fn, None)
                if eval_fn is not None:
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(os.path.join(args.output_dir, "results.jsonl"), "a") as f:
                        eval_fn_params = getattr(args, "eval_fn_params", None)
                        if eval_fn_params is None:
                            eval_fn_params = {}
                        metric = eval_fn(args.prediction_file_name, **eval_fn_params)
                        if len(metric) > 0:
                            metric.update({"file_name": args.prediction_file_name})
                            f.write(f"{json.dumps(metric)}\n")

        return super().on_predict(args, state, control, metrics, **kwargs)


class CausalFtTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        params: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        callback: Optional[LLMCallback] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # TODO: change the base class to Trainer
        self.combine_causal_loss_factor = params.pop("combine_causal_loss_factor", 0)
        self.only_causal_loss = params.pop("only_causal_loss", False)
        self.params = params
        self.extra_trace_dict = {}
        # self.optimizers = optimizers
        dist.barrier()
        # generation_config should be a generation_config instance or None, not a dict, so if eval_dataset is None, set this key to None
        if train_dataset and not eval_dataset:
            self.params["generation_config"] = None
        if eval_dataset:
            generation_config = GenerationConfig(**self.params.pop("generation_config"))
            self.params["generation_config"] = generation_config

        self.arguments = CausalFtTrainingArguments(**self.params)
        super().__init__(
            model=model,
            args=self.arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[LLMCallback] if not callback else None,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        train_sampler = self._get_train_sampler()
        data_collator = partial(data_collator, mode="train")
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self, eval_dataset: Dataset = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        eval_sampler = self._get_eval_sampler(eval_dataset)

        data_collator = partial(data_collator, mode="test")
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def preprocess_raw_labels(self, raw_labels):
        peft_config = self.get_peft_config()
        if peft_config:
            num_virtual_tokens = getattr(peft_config, "num_virtual_tokens", 0)
            if num_virtual_tokens > 0:
                prefix_labels = torch.full((raw_labels.size(0), num_virtual_tokens), -100).to(raw_labels.device)
                raw_labels = torch.cat((prefix_labels, raw_labels), dim=1)
        return raw_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_name = "label_loss"
        raw_labels = inputs.pop("raw_labels", None)
        if self.only_causal_loss:
            inputs["labels"] = raw_labels
            loss_name = "causal_loss"

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        self.state.losses = {loss_name: float(loss.cpu())}
        is_naslora = self.is_naslora()
        if is_naslora:
            # reg loss
            peft_config = self.get_peft_config()
            if peft_config.reg_loss:
                reg_loss = self.reg_loss()
                if reg_loss > 0:
                    loss += reg_loss
        if self.combine_causal_loss_factor > 0 and not self.only_causal_loss:
            raw_labels = self.preprocess_raw_labels(raw_labels)
            casual_loss_value = self.causal_loss(raw_labels, outputs.logits)
            self.state.losses.update({"causal_loss": float(casual_loss_value.cpu())})
            return loss + self.combine_causal_loss_factor * casual_loss_value
        else:
            return loss

    def reg_loss(self):
        if getattr(self, "choice_update_step", None):
            self.choice_update_step += 1
        else:
            self.choice_update_step = 1
        reg_loss = self.model.reg_loss()
        if reg_loss > 0:
            self.state.losses.update({"reg_loss": float(reg_loss.cpu())})

        is_naslora = self.is_naslora()
        step_size = self.choice_update_step
        if is_naslora:
            # reg loss
            peft_config = self.get_peft_config()
            step_size = getattr(peft_config, "step_size", self.choice_update_step)

        return reg_loss / ((self.choice_update_step // step_size + 1) * 200)

    def causal_loss(self, labels, logits):
        # labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size),
            shift_labels.view(batch_size * seq_length),
        )
        return loss

    def statisic(self) -> dict:
        """
        more statistics, any value you want to trace
        Returns
        -------
        dict
            more value
        """
        log_dict = {
            "total_flos": self.state.total_flos,
        }
        self.arch_parameters = getattr(self.model, "arch_parameters", [])
        if len(self.arch_parameters) > 0:
            log_dict["first_arch_param"] = self.arch_parameters[0].cpu().to(torch.float16).tolist()
            log_dict["last_arch_param"] = self.arch_parameters[-1].cpu().to(torch.float16).tolist()
        if getattr(self.state, "losses", None):
            log_dict.update(self.state.losses)
        search_step = int(self.state.max_steps / self.state.num_train_epochs)
        if getattr(self.model, "peft_config", None):
            if (
                getattr(
                    self.model.peft_config["default"],
                    "search_strategy",
                    "epoch",
                )
                == "step"
            ):
                # just for naslora serach step
                search_step = getattr(self.model.peft_config["default"], "search_step", 0)

        if search_step is not None and self.state.global_step < search_step + 5:
            log_dict.update(Model.count_pramameter(self.model, verbose=False))
        if search_step is not None and self.state.global_step < search_step:
            log_dict.update({"naslora_weights_lr": self.lr_scheduler._last_lr[-1]})
        return log_dict

    def trace_data(self, data: Dict[str, Union[int, float]]):
        """trace the data you want

        Parameters
        ----------
        data : Dict[str, Union[int, float]]
            the key/value you want to trace
        """
        if isinstance(data, dict):
            self.extra_trace_dict.update(data)
        else:
            assert isinstance(data, dict)

    def log(self, logs: Dict[str, float]) -> None:
        """call when log

        Parameters
        ----------
        logs : Dict[str, float]
            log information, such as {'loss':0.01, epoch: 0.12}

        Returns
        -------
        """
        logs.update(self.statisic())
        if isinstance(self.extra_trace_dict, dict):
            logs.update(self.extra_trace_dict)
        return super().log(logs)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        return super().save_model(output_dir, _internal_call)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)
        if dist.get_rank() == 0:
            # due to multiprocess, just do evaluation in rank 0
            metric = {}

            prediction_file_name = self.args.prediction_file_name
            eval_fn = getattr(metric_fn, getattr(self.args, "eval_fn", ""), None)

            if eval_fn is not None:
                eval_fn_params = getattr(self.args, "eval_fn_params", None)
                if eval_fn_params is None:
                    eval_fn_params = {}
                metric = eval_fn(prediction_file_name, **eval_fn_params)
                if len(metric) > 0:
                    self.log(metric)
        return results

    def is_naslora(self):
        is_naslora = False
        peft_config = getattr(self.model, "peft_config", None)
        if peft_config and peft_config["default"].peft_type.value == "NASLORA":
            is_naslora = True
        return is_naslora

    def get_peft_config(self):
        peft_config = getattr(self.model, "peft_config", None)
        if peft_config:
            peft_config = peft_config["default"]
        return peft_config

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        is_naslora = self.is_naslora()
        naslora_weights_lr = self.args.learning_rate
        if is_naslora:
            peft_config = self.get_peft_config()
            naslora_weights_lr = getattr(peft_config, "naslora_weights_lr")
            if naslora_weights_lr is None:
                naslora_weights_lr = self.args.learning_rate
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                        and (not is_naslora or "naslora_module" in n or "naslora_layernorms" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad) and (is_naslora and "naslora_weights" in n)
                    ],
                    "lr": naslora_weights_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        is_naslora = self.is_naslora()
        if_split = False
        milestones = 0
        if is_naslora:
            peft_config = self.get_peft_config()
            if peft_config.search_strategy == "step":
                milestones = peft_config.search_step
            else:
                milestones = num_training_steps // self.arguments.num_train_epochs
            step_size = peft_config.step_size
            gamma = peft_config.gamma
            if getattr(peft_config, "naslora_weights_lr"):
                if_split = True
        if self.lr_scheduler is None and (not is_naslora or not if_split):
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            self._created_lr_scheduler = True
        if self.lr_scheduler is None and is_naslora and if_split:
            # constant_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            #     self.optimizer if optimizer is None else optimizer,
            #     factor=0.5,
            #     total_iters=milestones,
            # )
            if peft_config.search_lr_scheduler == "cosine":
                constant_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer if optimizer is None else optimizer,
                    T_max=milestones,
                    # factor=1,
                    # total_iters=100,
                )
            elif peft_config.search_lr_scheduler == "steplr":
                constant_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer if optimizer is None else optimizer,
                    step_size=step_size,
                    gamma=gamma,
                    # factor=1,
                    # total_iters=100,
                )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer if optimizer is None else optimizer,
                T_max=num_training_steps,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer if optimizer is None else optimizer,
                schedulers=[constant_lr_scheduler, cosine_scheduler],
                milestones=[milestones],
            )
        return self.lr_scheduler

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        self.test_length = len(test_dataset)
        # for arnold
        if os.environ.get("ARNOLD_WORKER_GPU", None):
            self.gpu_number = int(os.environ.get("ARNOLD_WORKER_GPU", 1)) * int(os.environ.get("ARNOLD_WORKER_NUM", 1))
        else:
            self.gpu_number = torch.cuda.device_count()
        self.total_step = math.ceil(self.test_length // (self.params["per_device_eval_batch_size"] * self.gpu_number))
        self.current_step = 0
        return super().predict(test_dataset, ignore_keys, metric_key_prefix, **gen_kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        # logger.info(f"rank: {dist.get_rank()}-{inputs['input_ids'].size()}-{inputs['ground_truth_labels'].size()}")

        # extra ground truth data
        gd_truth = inputs.pop("ground_truth_labels", None)
        data_id = inputs.pop("data_id", None)
        if self.args.do_eval:
            inputs.pop("raw_labels", None)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}

        # extra params for generation_config
        self.params["generation_config"].bos_token_id = self.tokenizer.bos_token_id
        self.params["generation_config"].eos_token_id = self.tokenizer.eos_token_id
        self.params["generation_config"].pad_token_id = self.tokenizer.pad_token_id
        gen_kwargs["generation_config"] = self.params["generation_config"]

        scores = None
        # due to the max_length prioritization issues
        gen_kwargs.pop("max_length")
        if not getattr(self.params["generation_config"], "output_scores", False):
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)
        else:
            # set return_dict_in_generate to True in order to get the output dict
            self.params["generation_config"].return_dict_in_generate = True
            outputs = self.model.generate(**inputs, **gen_kwargs)
            generated_tokens, scores = outputs["sequences"], outputs["scores"]
        generated_str_length = generated_tokens.size(0)
        generated_token_probs = [[] for _ in range(generated_str_length)]
        if scores:
            # if having care tokens, we will output the target probs of care tokens for all generated positions
            care_token_ids = []
            if self.args.care_tokens:
                for token in self.args.care_tokens:
                    care_token_ids.append(self.tokenizer(token)["input_ids"][-1])

            generated_token_length = len(scores)

            for index_ in range(generated_token_length):
                target_scores = scores[index_]
                if len(care_token_ids) > 0:
                    target_scores = target_scores[:, care_token_ids]
                probs = (
                    torch.nn.functional.softmax(target_scores, dim=1).type(torch.float32).cpu().numpy()
                )  # bf16 should change type to torch.float32, so use torch.float32 to convert
                for _ in range(generated_str_length):
                    index_probs = generated_tokens[_][index_]
                    if care_token_ids:
                        generated_token_probs[_].append(probs[_])
                    else:
                        generated_token_probs[_].append(probs[_][index_probs])

            # for index_ in range(-generated_token_length, 0):
            #     probs = (
            #         torch.nn.functional.softmax(scores[index_], dim=1)
            #         .type(torch.float32)
            #         .cpu()
            #         .numpy()
            #     )  # bf16 should change type to torch.float32, so use torch.float32 to convert
            #     for _ in range(generated_str_length):
            #         index_probs = generated_tokens[_][index_]
            #         if self.args.care_tokens:
            #             index_probs = self.tokenizer.encode(
            #                 self.args.care_tokens
            #             )
            #         generated_token_probs[_].append(probs[_][index_probs])
            generated_token_probs = np.array(generated_token_probs).tolist()
            del scores

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        # process the generated tokens

        if dist.get_rank() == 0 and not self.args.do_eval:
            if self.current_step % self.params["logging_steps"] == 0:
                ground_truth_ = self.decode([gd_truth[0].cpu().numpy()]) if gd_truth is not None else None
                output = self.decode([generated_tokens[0].cpu().numpy()])
                logger.info(
                    f"[GPU_{dist.get_rank()}, {self.current_step}/{self.total_step}]: \n ground_truth: {ground_truth_[0]} \n output: {output[0]}"
                )
            self.current_step += 1
        self._output_generate_results(inputs, generated_tokens, gd_truth, generated_token_probs, data_id)
        generated_tokens = torch.tensor([1]).cuda(f"cuda:{dist.get_rank()}")
        # logger.info(f"rank: {dist.get_rank()}-{inputs['input_ids'].size()} over")
        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        return super()._pad_tensors_to_max_len(tensor, max_length)

    def decode(self, token_list):
        ignore_tokens = [
            self.tokenizer.bos_token,
            self.tokenizer.eos_token,
            self.tokenizer.pad_token,
            "\n",
            # "\x20",
        ]
        sub_re = re.compile("|".join(ignore_tokens))
        return list(map(lambda x: sub_re.sub("", self.tokenizer.decode(x)), token_list))

    def _output_generate_results(
        self,
        inputs: torch.Tensor,
        generated_tokens: torch.Tensor,
        gd_truth: Optional[torch.Tensor] = None,
        generated_token_probs: Optional[list] = None,
        data_id: torch.tensor = None,
    ):
        """output the greneted results to target file

        Parameters
        ----------
        inputs : torch.Tensor
            the inputs
        generated_tokens : torch.Tensor
            generated tokens
        gd_truth : Optional[torch.Tensor], optional
            the ground truth, by default None
        generated_token_probs: Optional[list], optional
            the prob, by default None
        data_id : torch.Tensor
            data id tokens
        Returns
        -------
        _type_
            _description_
        """
        data_id = [] if data_id is None else data_id
        input_length = inputs["input_ids"].size(1)

        generated_tokens = generated_tokens.detach().cpu()
        generated_q = generated_tokens[:, :input_length]
        generated_a = generated_tokens[:, input_length:]

        generated_q = generated_q.tolist()
        generated_a = generated_a.tolist()

        generated_tokens = list(
            map(
                lambda x: self.tokenizer.convert_ids_to_tokens(x)[: len(generated_token_probs[0])],
                generated_a,
            )
        )
        generated_tokens = list(
            map(
                lambda x: list(map(lambda y: y.replace(UNDERLINE, ""), x)),
                generated_tokens,
            )
        )

        generated_q_str = self.decode(generated_q)
        generated_a_str = self.decode(generated_a)

        gd_truth_str = ["Null"] * len(generated_a_str)
        if gd_truth is not None:
            gd_truth = gd_truth.cpu().tolist()
            gd_truth_str = self.decode(gd_truth)

        data_id_length = len(data_id)
        data_id_length = data_id_length if data_id_length else len(generated_a_str)
        data_id_str = ["Null"] * data_id_length
        if len(data_id) > 0:
            data_id = data_id.cpu().tolist()
            data_id_str = self.decode(data_id)
            data_id_str = list(map(lambda x: x.replace("id:/", ""), data_id_str))

        json_arr = []
        for question, label, label_g, tokens_g, probs, data_id_ in zip(
            generated_q_str,
            gd_truth_str,
            generated_a_str,
            generated_tokens,
            generated_token_probs,
            data_id_str,
        ):
            json_arr.append(
                dict(
                    question=question,
                    ground_truth=label,
                    answer=label_g,
                    generated_tokens=tokens_g,
                    probs=probs,
                    data_id=data_id_,
                )
            )

        self.jsonl_write(json_arr)

    def jsonl_write(self, json_arr: Dict[str, str]):
        """jsonl write

        Parameters
        ----------
        json_arr : Dict[str, str]
            json dict list
        output_file : str
            the output file
        """
        rank = dist.get_rank()
        json_str_arr = map(json.dumps, json_arr)
        output_file = self.args.prediction_file_name.replace(".jsonl", f"_{rank}.jsonl")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as writer:
            for line in json_str_arr:
                writer.write(f"{line}\n")


class SequenceClassificationTrainer(Trainer):
    def __init__(
        self,
        params: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        callback: Optional[LLMCallback] = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        # TODO: change the base class to Trainer
        self.params = params
        self.extra_trace_dict = {}
        # self.optimizers = optimizers
        dist.barrier()

        self.arguments = SequenceClassifyTrainingArguments(**self.params)
        
        # Trainer.__init__(
        super().__init__(
            model=model,
            args=self.arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[LLMCallback] if not callback else None,
            # callbacks=None,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            model_init=model_init,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
    # def create_optimizer(self):
    #     return super().create_optimizer()
        

TrainerMapping = {
    "Causal": CausalFtTrainer,
    "SequenceClassification": SequenceClassificationTrainer,
}
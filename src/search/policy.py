import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.trainer_callback import TrainerState

from ..peft.tuners.naslora import NASLoraModel
from ..peft.tuners.lora import LoraConfig
from ..utils.logging import get_logger, rank_zero_info

logger = get_logger("Search Policy")
rank_zero_info = partial(rank_zero_info, logger=logger)


class Search_Policy:
    _state_dict: TrainerState = None
    _model: NASLoraModel = None
    _peft_config: LoraConfig = None
    _active_status: bool = False
    _current_mode: int = -1
    _log: bool = False
    _patience_counter: int = 0
    _period: int = 0
    _totalization_loss: float = 0.0

    @classmethod
    def policy_handler(cls, state_dict: TrainerState, model: nn.Module, peft_config: LoraConfig) -> None:
        """a policy handler which is used to handle the policy

        Parameters
        ----------
        state_dict : TrainerState
            trainer state, contains loss, step, ...
        model : nn.Module
            a PeftModel
        peft_config : LoraConfig
            a peft config
        """
        cls._state_dict = state_dict
        cls._model = model
        cls._peft_config = peft_config

        # epoch [on_step_begin] = epoch [on_step_end] - 1, so, use < not <=
        global_step = (
            cls._state_dict.global_step if cls._peft_config.search_strategy == "step" else cls._state_dict.epoch
        )
        if global_step < cls._peft_config.search_step:
            cls._active_status = True
        else:
            cls._active_status = False

        search_fn = getattr(cls, cls._peft_config.search_policy, "None")
        search_fn_arguments = getattr(cls._peft_config, "policy_params", {})
        if cls._active_status and search_fn != "None":
            if not cls._log:
                rank_zero_info(f"Searching in space with {cls._peft_config.search_policy}")
                cls._log = True
            search_fn(**search_fn_arguments)
        else:
            ...

    @classmethod
    def decode(cls, mode: int):
        """
        mode: NASLoRA_modules weights | NASLoRA weights
                       0 -  unfreezed | unfreezed
                       1 -    freezed | unfreezed
                       2 -  unfreezed | freezed
                       3 -    freezed | freezed

        Parameters
        ----------
        mode : int
            0, 1, 2, 3
        """
        if cls._current_mode != mode:
            cls._model.lora_components_freezer(freeze_mode=mode)
            cls._current_mode = mode

    @classmethod
    def vanilla(cls, ratio: float) -> None:
        """just train lora modules' weights before total_step * ratio, and then just train lora module choice weights

        Parameters
        ----------
        ratio : float
            a thresholds
        """
        if cls._state_dict.global_step / cls._peft_config.search_step < ratio:
            cls.decode(2)
        if cls._state_dict.global_step / cls._peft_config.search_step > ratio:
            cls.decode(1)

    @classmethod
    def alternate(cls, ratio: float, interval: int = 2):
        """just train lora modules' weights before total_step * ratio, and then, update model weights alternately according to the `interval`

        Parameters
        ----------
        ratio : float
            a thresholds
        interval : int, optional
            a thresholds, when total_step % `interval` == 0, just train lora module choice weights, otherwise, just train lora modules' weights, by default 2
        """
        if cls._state_dict.global_step / cls._peft_config.search_step <= ratio:
            cls.decode(2)
        else:
            if cls._state_dict.global_step % interval == 0:
                cls.decode(1)
            else:
                cls.decode(2)

    @classmethod
    def independent(cls, step_1: int = 3, step_2: int = 3, pretrain_step: int = 0):
        if pretrain_step > cls._state_dict.global_step:
            cls.decode(2)
        else:
            local_step = (cls._state_dict.global_step - pretrain_step) % (step_1 + step_2)
            if local_step < step_1 - 1:
                cls.decode(2)
            else:
                cls.decode(1)

    @classmethod
    def adaptive(
        cls,
        patience_1: int = 5,
        patience_2: int = 5,
        threshold_1: float = 0.1,
        threshold_2: float = 0.1,
    ):
        def change():
            if cls._state_dict.global_step % 2 == 0:
                cls.decode(1)
            else:
                cls.decode(2)

        best_min_loss = getattr(cls, "_best_min_loss", 10000)

        loss_arr = list(
            map(
                lambda x: round(x["loss"], 3),
                cls._state_dict.log_history,
            )
        )

        # if we do pretrain, the patience may be larger
        if cls._period == 0:
            _threshold = threshold_1
            _patience = patience_1
        else:
            _threshold = threshold_2
            _patience = patience_2

        if len(loss_arr) > 0:
            current_loss = loss_arr[-1]
            if current_loss < best_min_loss:
                cls._totalization_loss += best_min_loss - current_loss

                cls._best_min_loss = current_loss
                if cls._totalization_loss > _threshold:
                    cls._patience_counter = 0
                    cls._totalization_loss = 0
                else:
                    cls._patience_counter += 1
            else:
                cls._patience_counter += 1

            if cls._patience_counter > _patience:
                cls._period += 1
                cls._patience_counter = 0
                cls._totalization_loss = 0

        if cls._period == 0:
            cls.decode(2)
        elif cls._period == 1:
            change()
        else:
            cls._model.peft_config["default"].search_step = cls._state_dict.global_step + 1
            change()

    @classmethod
    def pretrained_combine(cls, pretrain_step: int = 0):
        if pretrain_step > cls._state_dict.global_step:
            cls.decode(2)
        else:
            cls.decode(0)

    @classmethod
    def pretrained_combine_adaptive(cls, patience: int = 10):
        if cls._patience_counter < patience:
            best_min_loss = getattr(cls, "_best_min_loss", 10000)
            current_loss = (
                10000 if len(cls._state_dict.log_history) == 0 else round(cls._state_dict.log_history[-1]["loss"], 3)
            )
            if current_loss / best_min_loss > 0.5:
                cls._patience_counter += 1
                cls.decode(2)
            else:
                cls._best_min_loss = current_loss
                cls._patience_counter = 0
        else:
            cls.decode(0)

        # if pretrain_step > cls._state_dict.global_step:
        #     best_min_loss = getattr(cls, "_best_min_loss", 10000)
        #     cls.decode(2)
        # else:
        #     best_min_loss = getattr(cls, "_best_min_loss", 10000)
        #     try:
        #         loss_arr = list(
        #             map(
        #                 lambda x: round(x["loss"], 3),
        #                 cls._state_dict.log_history,
        #             )
        #         )
        #         if len(loss_arr) > 0:
        #             current_loss = loss_arr[-1]
        #             if best_min_loss > current_loss:
        #                 best_min_loss = current_loss
        #                 cls._patience_counter = 0
        #                 cls.decode(0)
        #             else:
        #                 cls._patience_counter += 1
        #             if cls._patience_counter >= patience:
        #                 cls._model.peft_config["default"].search_step = (
        #                     cls._state_dict.global_step + 1
        #                 )
        #                 cls.decode(2)
        #     except:
        #         cls._model.peft_config["default"].search_step = (
        #             cls._state_dict.global_step + 1
        #         )
        #         cls.decode(2)

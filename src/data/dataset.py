import json
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import datasets
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import (
    SequentialDistributedSampler,
    distributed_concat,
)
import evaluate
import numpy as np
from transformers import PretrainedConfig, default_data_collator, DataCollatorWithPadding

from ..utils.logging import get_logger
from functools import partial
datasets.load_dataset = partial(datasets.load_dataset, trust_remote_code=True)
IGNORE_INDEX = -100

logger = get_logger("Dataset")


class CommonDataset:
    def __init__(self, params) -> None:
        self.params = params

    def post_process(self, **args):
        ...
    
    def get_datset_type(self, dataset_path: str):
        if ".json" in dataset_path:
            return "json"
        elif ".csv" in dataset_path:
            return "csv"
        elif ".parquet" in dataset_path:
            return "parquet"
        elif ".arrow" in dataset_path:
            return "arrow"
        else:
            raise ValueError(f"{dataset_path} not support")

    def get_train_dataset(self, size: int = -1):
        filename = self.params["train_file"]
        dataset_type = self.get_datset_type(filename)
        train_dataset = datasets.load_dataset(
            dataset_type, data_files=filename
        )["train"]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        filename = self.params["test_file"]
        dataset_type = self.get_datset_type(filename)
        test_dataset = datasets.load_dataset(
            dataset_type, data_files={"test": filename}
        )["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        filename = self.params["eval_file"]
        dataset_type = self.get_datset_type(filename)
        test_dataset = datasets.load_dataset(
            dataset_type, data_files={"eval": filename}
        )["eval"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def _padding_all(
        self, batch_pad_truncate: bool = True, mode: str = "train", **kwargs
    ): ...

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(b["instruction"])
                + tokenizer.tokenize(b["input"])
            )
            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(b["output"])
            )
            ground_truths.append(gd_truth_tokens)
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )
            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(b["output"]) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len
        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][:padding_length]
                    attention_mask[c] = attention_mask[c][:padding_length]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][:padding_length]
                    attention_mask[c] = attention_mask[c][:padding_length]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        data = {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }
        data.update(self.identify_property(batch, tokenizer))
        return data

    def identify_property(self, batch: List[Dict[str, Any]], tokenizer):
        max_length = 200

        def extract_information(data):
            id_str = "id:"
            for key, value in data.items():
                if "id" in key:
                    id_str += f"/{value}"
            tokens = tokenizer.tokenize(id_str)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids = ids[:max_length]
            ids.extend([tokenizer.pad_token_id] * (max_length - len(ids)))
            return ids

        ids = torch.tensor(list(map(extract_information, batch)))
        return dict(data_id=ids)

    def metric(self, eval_preds): ...

    def get_dataloader(self):
        sampler = None

    @classmethod
    def get_distribution(cls, dataset: datasets.Dataset):
        import matplotlib.pyplot as plt
        import seaborn as sns

        dataset_name = dataset.builder_name
        output_dir = f"./dataset_distributions/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        features = dataset.features
        dataset_sample_legnth = {_: [] for _ in features}
        for sampler in dataset:
            for feature in features:
                dataset_sample_legnth[feature].append(
                    len(str(sampler[feature]))
                )
        for feature in features:
            sns.distplot(dataset_sample_legnth[feature], color="m", kde=True)
            plt.savefig(f"{output_dir}/dist_{feature}.png")
            plt.close()

        logger.info(
            f"The distributions of the dataset `{dataset_name}` have been generated in `{output_dir}`, please check ~"
        )

        ...


class ExternalPlatformDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def metric(self, eval_preds):
        x = 1
        ...

    def collate_fn(
        self,
        batch,
        tokenizer=None,
        max_len: int = None,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            current_input_tokens = [tokenizer.bos_token] + tokenizer.tokenize(
                b["question"]
            )
            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )

            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(b["answer"])
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = (
                    [tokenizer.bos_token]
                    + tokenizer.tokenize(b["answer"])
                    + [tokenizer.eos_token]
                )
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len
        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]

        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class CCRDataset(ExternalPlatformDataset):
    def get_train_dataset(self, size: int = -1):
        return super().get_train_dataset(size)

    def get_eval_dataset(self, size: int = -1):
        return super().get_eval_dataset(size)

    def collate_fn(
        self,
        batch,
        tokenizer=None,
        max_len: int = None,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            current_input_tokens = tokenizer.tokenize(b["question"])
            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )

            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(b["answer"])
            )
            ground_truths.append(gd_truth_tokens)
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )
            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(b["answer"]) + ["[EOS]"]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len
        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]

        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class BoolqDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"].split("-")
        train_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "train"
        ]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"].split("-")
        test_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "test"
        ]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"].split("-")
        val_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "validation"
        ]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            contents = b["passage"].split(" -- ")
            subject = contents[0]
            article = " -- ".join(contents[1:])

            question = b["question"]
            answer = "True" if b["label"] == 1 else "False"
            instruct = "Read the following article and answer the question, with True and False."

            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Article: "' + article + '"')
                + tokenizer.tokenize(' Question: "' + subject + question + '"')
                + tokenizer.tokenize(" Answer: ")
            )

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class SIQADataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"]
        train_dataset = datasets.load_dataset(dataset_name)["train"]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        filename = self.params["test_file"]
        test_dataset = datasets.load_dataset(
            "jsonl", data_files={"test": filename}
        )["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"]
        val_dataset = datasets.load_dataset(dataset_name)["validation"]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = "Read the following text material and choose a best answer to the question."

            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Context: "' + b["context"] + '"')
                + tokenizer.tokenize(' Question: "' + b["question"] + '"')
                + tokenizer.tokenize(' answer a: "' + b["answerA"] + '"')
                + tokenizer.tokenize('; answer b: "' + b["answerB"] + '"')
                + tokenizer.tokenize('; answer c: "' + b["answerC"] + '"')
                + tokenizer.tokenize(". Your choice would be: ")
            )

            answer = (
                "a" if b["label"] == "1" else "b" if b["label"] == "2" else "c"
            )
            # if b['label'] == '1': answer = 'A'
            # elif b['label'] == '2': answer = 'B'
            # else: answer = 'C'

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class RecordDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"].split("-")
        train_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "train"
        ]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"].split("-")
        test_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "test"
        ]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"].split("-")
        val_dataset = datasets.load_dataset(dataset_name[0], dataset_name[1])[
            "validation"
        ]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = (
                "Read the following text material and choose entities to fill in the blank of query. "
                + "Please follow the instructions: "
                + '1) The related entites in the texts would be identified with double equal marks, "==", at the begining and end of entities; '
                + '2) After reading the article, some key points would be given after the symbol "@highlight"; '
                + '3) The blank of the query would stressed with "@placeholder". '
            )

            sent = b["passage"]
            indexes = b["entity_spans"]
            cnt = 0
            for s, e in zip(indexes["start"], indexes["end"]):
                s, e = s + 4 * cnt, e + 4 * cnt
                sent = sent[:s] + "==" + sent[s:e] + "==" + sent[e:]
                cnt += 1

            # sents = sent.split('\n@highlight\n')
            # passage = sents[0]
            # highlights = ''
            # for i in range(len(sents[1:])):
            #     highlights += f'\nHighlight {i}:' + sents[i+1]
            # sent = passage + highlights

            pattern = (
                instruct
                + "\n"
                + "Context: "
                + sent
                + "\n"
                + "Related entities: "
                + ", ".join(b["entities"])
                + "\n"
                + "Query: "
                + b["query"]
                + "\n"
                + 'Now pick just one entity to fill in place of "@placeholder" in the given query. Which entity would be the best?'
                + "\n"
                + "Answer: "
            )
            if len(b["answers"]) > 1:
                pattern += (
                    "(the following multiple answers are equally correct)"
                )
                answer = ", ".join(b["answers"])
            else:
                answer = b["answers"][0]

            # logger.info(pattern)
            # logger.info('answer: ', answer, type(answer))
            # logger.info('ori answer:', b['answers'], len(b['answers']), type(b['answers']))

            current_input_tokens = [tokenizer.bos_token] + tokenizer.tokenize(
                pattern
            )

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class CNNDailymailDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"]
        train_dataset = datasets.load_dataset(dataset_name, "3.0.0")["train"]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"]
        test_dataset = datasets.load_dataset(dataset_name, "3.0.0")["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"]
        val_dataset = datasets.load_dataset(dataset_name, "3.0.0")["validation"]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = "Read the following article and then please summarize the highlights of the article. \n"

            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Article: "' + b["article"] + '"')
                + tokenizer.tokenize(".\n The highlights are: ")
            )

            answer = b["highlights"]

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class SamSumDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"]
        train_dataset = datasets.load_dataset(dataset_name)["train"]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"]
        test_dataset = datasets.load_dataset(dataset_name)["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"]
        # val_dataset = datasets.load_dataset(dataset_name)["validation"]
        val_dataset = datasets.load_dataset(dataset_name)["test"]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = "Read the following dialogue and then please summarize this dialogue. \n"

            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Article: "' + b["dialogue"] + '"')
                + tokenizer.tokenize(".\n The summary is: ")
            )

            answer = b["summary"]

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class GSM8KDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"]
        train_dataset = datasets.load_dataset(dataset_name, "main")["train"]
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"]
        test_dataset = datasets.load_dataset(dataset_name, "main")["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"]
        # val_dataset = datasets.load_dataset(dataset_name)["validation"]
        val_dataset = datasets.load_dataset(dataset_name, "main")["test"]
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = (
                "Please solve the following math question step by step.\n"
            )
            # cot = b["answer"].split("####")[0]
            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Question: "' + b["question"] + '"')
                + tokenizer.tokenize(".\n Your answer is: ")
            )

            # answer = re.findall(r"\d+", b["answer"].split("####")[-1])
            answer_raw = b["answer"].split("####")
            answer = (
                answer_raw[0] + "\n Therefore, the result is:" + answer_raw[-1]
            )

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class XSumDataset(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        dataset_name = self.params["train_file"]
        train_dataset = datasets.load_dataset(dataset_name)["train"]
        # train_dataset = train_dataset.filter(
        #     lambda x: len(x["document"].split(" ")) < max_length
        # )
        if size > 0:
            train_dataset = train_dataset.shuffle(seed=42)
        if size == -1:
            return train_dataset
        else:
            return train_dataset.select(range(size))

    def get_test_dataset(self, size: int = -1):
        dataset_name = self.params["test_file"]
        test_dataset = datasets.load_dataset(dataset_name)["test"]
        if size == -1:
            return test_dataset
        else:
            return test_dataset.select(range(size))

    def get_eval_dataset(self, size: int = -1):
        dataset_name = self.params["eval_file"]
        # val_dataset = datasets.load_dataset(dataset_name)["validation"]
        val_dataset = datasets.load_dataset(dataset_name)["validation"]
        if size > 0:
            val_dataset = val_dataset.shuffle(seed=42)
        if size == -1:
            return val_dataset
        else:
            return val_dataset.select(range(size))

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        (
            inputs,
            attention_mask,
            labels,
            raw_labels,
            ground_truths,
            context_end_index,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # input process
            instruct = "Read the following document and then please summarize this document. \n"
            # cot = b["answer"].split("####")[0]
            current_input_tokens = (
                [tokenizer.bos_token]
                + tokenizer.tokenize(instruct)
                + tokenizer.tokenize(' Document: "' + b["document"])
            )
            context_end_index.append(len(current_input_tokens))
            current_input_tokens += tokenizer.tokenize(
                '"' + ".\n The summary is: "
            )

            # answer = re.findall(r"\d+", b["answer"].split("####")[-1])
            answer = b["summary"]

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    exceed_length = len(inputs[c]) - padding_length
                    context_end_index_ = context_end_index[c] - exceed_length
                    inputs[c] = (
                        inputs[c][:context_end_index_]
                        + inputs[c][context_end_index[c] :]
                    )
                    labels[c] = (
                        labels[c][:context_end_index_]
                        + labels[c][context_end_index[c] :]
                    )
                    raw_labels[c] = (
                        raw_labels[c][:context_end_index_]
                        + raw_labels[c][context_end_index[c] :]
                    )
                    attention_mask[c] = (
                        attention_mask[c][:context_end_index_]
                        + attention_mask[c][context_end_index[c] :]
                    )
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    exceed_length = len(inputs[c]) - padding_length
                    context_end_index_ = context_end_index[c] - exceed_length
                    inputs[c] = (
                        inputs[c][:context_end_index_]
                        + inputs[c][context_end_index[c] :]
                    )
                    attention_mask[c] = (
                        attention_mask[c][:context_end_index_]
                        + attention_mask[c][context_end_index[c] :]
                    )
        if not ignore_q:
            labels = raw_labels
        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


class SuperGlueDatasets(CommonDataset):
    def __init__(self, params) -> None:
        super().__init__(params)

    def get_train_dataset(self, size: int = -1):
        ttl_dt = self.params["train_file"]
        # ttl_dt = ['boolq', 'cb', 'copa', 'rte', 'wic', 'wsc', 'multirc', 'record']
        res = []
        for dt in ttl_dt:
            train_dataset = datasets.load_dataset("super_glue", dt)["train"]
            train_dataset = train_dataset.shuffle(seed=42)
            max_length = min(len(train_dataset), 10000)
            train_dataset = train_dataset.select(range(max_length))
            if dt == "record":
                new_set = {k: [] for k in list(train_dataset.features)}
                for item in tqdm.tqdm(train_dataset):
                    if len(item["answers"]) == 1:
                        for k in list(train_dataset.features):
                            new_set[k].append(item[k])
                    else:
                        for a in item["answers"]:
                            for k in list(train_dataset.features):
                                if k != "answers":
                                    new_set[k].append(item[k])
                                if k == "answers":
                                    new_set[k].append([a])
                train_dataset = datasets.Dataset.from_dict(
                    new_set, features=train_dataset.features
                )
                del new_set

            processor = getattr(self, dt, None)
            if processor:
                processor = partial(processor, mode="train")
                ft = train_dataset.features
                new_df = train_dataset.map(
                    processor, remove_columns=list(ft), batched=False
                )
                res.append(new_df)
        res = datasets.concatenate_datasets(res)
        if size > 0:
            res = res.shuffle(seed=42)
        if size == -1:
            return res
        else:
            return res.select(range(size))

    def get_test_dataset(self, size: int = -1):
        ttl_dt = self.params["test_file"]
        # ttl_dt = ['boolq', 'cb', 'copa', 'rte', 'wic', 'wsc', 'multirc', 'record']
        res = []
        for dt in ttl_dt:
            df_hf = datasets.load_dataset("super_glue", dt)["test"]
            processor = getattr(self, dt, None)
            if processor:
                self.dataset_process(res, df_hf, processor, dt)
        res = datasets.concatenate_datasets(res)
        if size == -1:
            return res
        else:
            return res.select(range(size))

    def prefix(self, item, dt):
        """for test

        Parameters
        ----------
        item : _type_
            _description_
        dt : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        idx = {"idx": item["idx"]}
        return f"<idx>{json.dumps(idx)}</idx><l>{dt}</l>{item['answer']}"

    def dataset_process(self, res, df_hf, processor, dt):
        processor = partial(processor, mode="eval")
        ft = df_hf.features
        ft.pop("idx")
        new_df = df_hf.map(processor, remove_columns=list(ft), batched=False)
        new_df = new_df.map(
            lambda x: {
                "question": x["question"],
                "answer": f"{self.prefix(x, dt)}",
            },
            remove_columns=["idx"],
            batched=False,
        )
        res.append(new_df)

    def get_eval_dataset(self, size: int = -1):
        ttl_dt = self.params["eval_file"]
        # ttl_dt = ['boolq', 'cb', 'copa', 'rte', 'wic', 'wsc', 'multirc', 'record']
        res = []

        for dt in ttl_dt:
            df_hf = datasets.load_dataset("super_glue", dt)["validation"]
            processor = getattr(self, dt, None)
            if processor:
                self.dataset_process(res, df_hf, processor, dt)

        res = datasets.concatenate_datasets(res)
        if size > 0:
            res = res.shuffle(seed=42)
        if size == -1:
            return res
        else:
            return res.select(range(size))

    def collate_handler(
        self, origin: str, entry: Dict[str, Any], mode: str = "train"
    ):
        """to assign collate functions for data coming from different origins.
        The collate functions returns 2 parts of input: question and answer
        """
        processor = getattr(self, origin, None)
        if not processor:
            raise AttributeError("Did not find any suitable collate function")
        return processor(entry, mode)

    def boolq(self, entry: Dict[str, Any], mode="train"):
        """BoolQ: 判断给定问题的答案是否为“是”或“否”"""
        contents = entry["passage"].split(" -- ")
        subject = contents[0]
        article = " -- ".join(contents[1:])

        input_question = entry["question"]
        answer = "True" if entry["label"] == 1 else "False"
        instruct = "Read the following article and answer the question, with True and False."

        question = (
            instruct
            + ' Article: "'
            + article
            + '"'
            + ' Question: "'
            + subject
            + input_question
            + '"'
            + " Answer: "
        )
        return {"question": question, "answer": answer}

    def cb(self, entry: Dict[str, Any], mode="train"):
        """Commitment Bank: 一段包含嵌入子句的前提，以及该子句成立的假设，每条嵌入语句都标有撰写文本的人对真实性的承诺程度"""
        premise = entry["premise"]
        hypo = entry["hypothesis"]
        answer = (
            "Entailment"
            if entry["label"] == 0
            else "Contradiction" if entry["label"] == 1 else "Neutral"
        )
        instruct = "Read the following texts. The first sentence is premise, the second sentence is hypothesis to the premise. Then decide the relationship of these two sentences."

        question = (
            instruct
            + ' Premise: "'
            + premise
            + '"'
            + ' Hypothesis: "'
            + hypo
            + '"'
            + " So what is the relationship of these two sentences? Entailment, contradiction, or neutral? Answer: "
        )
        return {"question": question, "answer": answer}

    def copa(self, entry: Dict[str, Any], mode="train"):
        """Choice of Plausible Alternatives: 因果推理任务, 一个前提语句，并且必须从两个可能的选择中确定前提的原因或结果"""
        premise = entry["premise"]
        c1 = entry["choice1"]
        c2 = entry["choice2"]
        answer = "1" if entry["label"] == 0 else "2"
        instruct = "Read the following texts. The first sentence is premise. The following two sentences, act as possible cause or effect of the premise. Follow the instruction to find the right cause/effect to the premise."

        question = (
            instruct
            + ' Premise: "'
            + premise
            + '"'
            + ' Sentence 1: "'
            + c1
            + '"'
            + ' Sentence 2: "'
            + c2
            + '"'
            + " Which is the "
            + entry["question"]
            + " to the premise? Use 1 or 2 to answer the question. Answer:"
        )

        return {"question": question, "answer": answer}

    def rte(self, entry: Dict[str, Any], mode="train"):
        """Recognizing Textual Entailment: 判断给定的两个句子是否具有蕴含关系"""
        premise = entry["premise"]
        hypo = entry["hypothesis"]
        answer = "Yes" if entry["label"] == 0 else "No"
        instruct = "The following sentences consist of a combination of premise and hypothesis. What is the relationship of the two sentences? "

        question = (
            instruct
            + ' Premise: "'
            + premise
            + '"'
            + ' Hypothesis: "'
            + hypo
            + '"'
            + " Does the hypothesis entails the premise or not? Answer: "
        )

        return {"question": question, "answer": answer}

    def wic(self, entry: Dict[str, Any], mode="train"):
        """Word-in-Context: 词义消歧, 给定两个文本片段和一个出现在两个句子中的多义词，任务是确定该词在两个句子中是否是相同的含义"""
        word = entry["word"]
        answer = "True" if entry["label"] == 1 else "False"
        instruct = (
            "Does the given word share the similar meaning in the two different sentences? "
            + 'The given word in the sentences is highlighted with double equal marks, "==", at the begining and end of the word.'
        )

        sent1 = (
            entry["sentence1"][: entry["start1"]]
            + "=="
            + entry["sentence1"][entry["start1"] : entry["end1"]]
            + "=="
            + entry["sentence1"][entry["end1"] :]
        )
        sent2 = (
            entry["sentence2"][: entry["start2"]]
            + "=="
            + entry["sentence2"][entry["start2"] : entry["end2"]]
            + "=="
            + entry["sentence2"][entry["end2"] :]
        )
        question = (
            instruct
            + ' Sentence 1: "'
            + sent1
            + '"'
            + ' Sentence 2: "'
            + sent2
            + '"'
            + ' Does the given word "'
            + word
            + '" has similar meaning in the sentences or not? Answer: '
        )

        return {"question": question, "answer": answer}

    def wsc(self, entry: Dict[str, Any], mode="train"):
        """Winograd Schema Challenge: 指代消歧，任务是确定代词是否指代该名词"""
        text = entry["text"].split(" ")
        text[entry["span1_index"]] = "==" + text[entry["span1_index"]] + "=="
        text[entry["span2_index"]] = "==" + text[entry["span2_index"]] + "=="
        sent = " ".join(text)

        instruct = "Does the pronoun have the same referent as the noun in the sentence?"
        question = (
            instruct
            + ' Sentence: "'
            + sent
            + '"'
            + ' Word 1: "'
            + entry["span1_text"]
            + '"'
            + ' Word 2: "'
            + entry["span2_text"]
            + '"'
            + ' So does the two words point to the same thing or not? Please answer the question with "True" or "False". Answer: '
        )

        answer = "True" if entry["label"] == 1 else "False"

        return {"question": question, "answer": answer}

    def multirc(self, entry: Dict[str, Any], mode="train"):
        """Multi-Sentence Reading Comprehension: QA 任务, 给出文章、问题与可能回答，判断答案是否正确"""
        instruct = "Read the passage and decide if the response to the question is correct or not."
        question = (
            instruct
            + " Passage: "
            + entry["paragraph"]
            + "\n"
            + " Question: "
            + entry["question"]
            + "\n"
            + " Response: "
            + entry["answer"]
            + "\n"
            + ' Now use "True" or "False" to decide if the response to the question is correct or not. Answer: '
        )
        answer = "True" if entry["label"] == 1 else "False"

        return {"question": question, "answer": answer}

    def record(self, entry: Dict[str, Any], mode="train"):
        instruct = (
            "Read the following text material and choose entities to fill in the blank of query."
            + "Please follow the instructions: "
            + ' 1) The related entites in the texts would be identified with double equal marks, "==", at the begining and end of entities;'
            + ' 2) After reading the article, some key points would be given after the symbol "@highlight";'
            + ' 3) The blank of the query would stressed with "@placeholder". '
        )
        sent = entry["passage"]
        indexes = entry["entity_spans"]
        cnt = 0
        for s, e in zip(indexes["start"], indexes["end"]):
            s, e = s + 4 * cnt, e + 4 * cnt
            sent = sent[:s] + "==" + sent[s:e] + "==" + sent[e:]
            cnt += 1

        pattern = (
            instruct
            + "\n"
            + "Context: "
            + sent
            + "\n"
            + "Related entities: "
            + ", ".join(entry["entities"])
            + "\n"
            + "Query: "
            + entry["query"]
            + "\n"
            + 'Now pick just one entity to fill in place of "@placeholder" in the given query. Which entity would be the best?'
            + "\n"
            + "Answer: "
        )

        if mode == "train":
            answer = entry["answers"][0]
        else:
            if len(entry["answers"]) > 1:
                answer = ", ".join(entry["answers"])
            else:
                answer = entry["answers"][0] if entry["answers"] else ""

        return {"question": pattern, "answer": answer}

    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # if 'origin' not in b:
            #     logger.info('no origin found for this data point...')
            #     logger.info(b)
            #     origin = 'record'
            # else: origin = b.pop('origin')
            # origin_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(origin))
            question, answer = b["question"], b["answer"]

            current_input_tokens = [tokenizer.bos_token] + tokenizer.tokenize(
                question
            )

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels

        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }


# GLUE datasets
class GLUEDataset_Common(CommonDataset):
    dataset_name = "nyu-mll/glue"
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    def __init__(self, params):
        super().__init__(params)
        self.get_raw_datasets()
    
    def get_raw_datasets(self):
        self.raw_datasets = datasets.load_dataset(
            self.dataset_name,
            self.params["train_file"],
            cache_dir=self.params.get("cache_dir", None),
            token=self.params.get("token", None),
        )
        # Labels    
        self.is_regression = self.params["train_file"] == "stsb"
        if not self.is_regression:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1
        
        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = self.task_to_keys[self.params["train_file"]]
        
        # Padding strategy
        if self.params.get("pad_to_max_length", False):
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False
    def post_process(self, **args):
        
        model = args.get("model", None)
        config = args.get("config", None)
        tokenizer = args.get("tokenizer", None)
        if model is None:
            ...
        self.label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=self.num_labels).label2id
            and self.params["train_file"] is not None
            and not self.is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if sorted(label_name_to_id.keys()) == sorted(self.label_list):
                self.label_to_id = {i: int(label_name_to_id[self.label_list[i]]) for i in range(self.num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(self.label_list)}."
                    "\nIgnoring the model labels as a result.",
                )
        elif self.params["train_file"] is None and not self.is_regression:
            self.label_to_id = {v: i for i, v in enumerate(self.label_list)}
        
        if self.label_to_id is not None:
            model.config.label2id = self.label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif self.params["train_file"] is not None and not self.is_regression:
            model.config.label2id = {l: i for i, l in enumerate(self.label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        
        self.max_seq_length = min(self.params["max_seq_length"], tokenizer.model_max_length)
        
        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
            )
            result = tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if self.label_to_id is not None and "label" in examples:
                result["label"] = [(self.label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result
        self.raw_datasets = self.raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not self.params.get("overwrite_cache", False),
            desc="Running tokenizer on dataset",
        )
        
        # get collate_fn
        if self.params.get("pad_to_max_length", False):
            self.data_collator = default_data_collator
        elif self.params.get("fp16", False):
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            self.data_collator = None
        
        # get metric function
        # Get the metric function
        if self.params["train_file"] is not None:
            self.metric_ = evaluate.load("glue", self.params["train_file"], cache_dir=self.params["cache_dir"])
        elif self.is_regression:
            self.metric_ = evaluate.load("mse", cache_dir=self.params["cache_dir"])
        else:
            self.metric_ = evaluate.load("accuracy", cache_dir=self.params["cache_dir"])
        
    def get_train_dataset(self, size=-1):
        train_dataset = self.raw_datasets["train"]
        if size != -1:
            max_train_samples = min(len(train_dataset), size)
            train_dataset = train_dataset.select(range(max_train_samples))
        return train_dataset
        
    def get_test_dataset(self, size: int = -1):
        return super().get_test_dataset(size)
    def get_eval_dataset(self, size: int = -1):
        eval_dataset = self.raw_datasets["validation_matched" if self.params["eval_file"] == "mnli" else "validation"]
        if size != -1:
            max_eval_samples = min(len(eval_dataset), size)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        return eval_dataset
    
    def collate_fn(self, *arg, **args):
        return self.data_collator(*arg, **args)
    
    def metric(self, p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        result = self.metric_.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result


# GLUE LLM datasets
class GLUEDataset_LLM(CommonDataset):
    dataset_name = "nyu-mll/glue"
    instructs = {
        "cola": "Read the following sentece and decide if it is grammatically correct or not.",
        "sst2": "Read the following sentence and decide if it is a positive sentiment or not.",
        "mrpc": "Read the following two sentences and decide if they have the same meaning or not.",
        "stsb": "Read the following two sentences and rate their semantic similarity on a scale of 1 to 5.",
        "qqp": "Read the following two questions and decide if they have the same meaning or not.",
        "mnli": "Read the following premise and hypothesis and decide if the premise entails the hypothesis, contradicts it, or neither entails nor contradicts it.",
        "qnli": "Read the following question and decide if the answer is correct or not.",
        "rte": "Read the following two sentences and decide if the second sentence is entailed by the first sentence.",  
        "wnli": "Read the following two sentences and decide if the second sentence is entailed by the first sentence.",
    }
    def __init__(self, params):
        super().__init__(params)
    
    def cola(self, entry: Dict[str, Any], mode="train"):
        """cola: 语言可接受性语料库包括从语言学理论方面的书籍和期刊文章中提取的英语可接受性判断。每个例子都是一个单词序列，并注释它是否是一个合乎语法的英语句子。"""
        dt = "cola"
        sentence = entry["sentence"]
        answer = "True" if entry["label"] == 1 else "False"
        question = f"""{self.instructs[dt]} 
 Sentence: '{sentence}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def sst2(self, entry: Dict[str, Any], mode="train"):
        """sst2: 斯坦福情感树库由电影评论中的句子和人类对其情感的注释组成。任务是预测给定句子的情绪。它使用双向(正/负)类拆分，只使用句子级别的标签。"""
        dt = "sst2"
        sentence = entry["sentence"]
        answer = "True" if entry["label"] == 1 else "False"
        question = f"""{self.instructs[dt]}. 
 Sentence: '{sentence}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def mrpc(self, entry: Dict[str, Any], mode="train"):
        """mrpc: 微软研究释义语料库(Dolan & Brockett, 2005)是从在线新闻来源自动提取的句子对语料库，对句子对中的句子是否语义等同进行人工注释。"""
        dt = "mrpc"
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = "True" if entry["label"] == 1 else "False"
        question = f"""{self.instructs[dt]}. 
 Sentence1: '{sentence1}'
 Sentence2: '{sentence2}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def stsb(self, entry: Dict[str, Any], mode="train"):
        """stsb: 语义文本相似基准(Cer et al.， 2017)是从新闻标题、视频和图像标题以及自然语言推理数据中提取的句子对的集合。每一对都是人工标注的，相似度评分从1到5。"""
        dt = "stsb"
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = f"{round(entry['label'], 3)}"
        question = f"""{self.instructs[dt]}. 
 Sentence1: '{sentence1}'
 Sentence2: '{sentence2}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def qqp(self, entry: Dict[str, Any], mode="train"):
        """qqp: Quora问题对2数据集是来自社区问答网站Quora的问题对集合。任务是确定一对问题在语义上是否相等。"""
        dt = "qqp"
        question1 = entry["question1"]
        question2 = entry["question2"]
        answer = "True" if entry["label"] == 1 else "False"
        question = f"""{self.instructs[dt]}. 
 Question1: '{question1}'
 Question2: '{question2}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def mnli(self, entry: Dict[str, Any], mode="train"):
        """多体裁自然语言推理语料库是一个带有文本蕴涵注释的句子对的众包集合。给定一个前提句和一个假设句，任务是预测前提是否包含假设(蕴涵)，与假设(矛盾)相矛盾，还是两者都不包含(中立)。这些前提句来自十种不同的来源，包括录音、小说和政府报告。基准测试的作者使用标准测试集，他们从RTE作者那里获得了私有标签，并在两个方面进行评估"""
        dt = "mnli"
        premise = entry["premise"]
        hypothesis = entry["hypothesis"]
        answer = "neutral"
        if entry["label"] == 0:
            answer = "entailment"
        elif entry["label"] == 2:
            answer = "contradiction"
        question = f"""{self.instructs[dt]}. 
 Premise: '{premise}'
 Hypothesis:'{hypothesis}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def qnli(self, entry: Dict[str, Any], mode="train"):
        """qnli: 斯坦福问答数据集是一个问答数据集，由问题-段落对组成，其中段落中的一个句子(取自维基百科)包含相应问题的答案(由注释者编写)。基准测试的作者通过在每个问题和对应上下文中的每个句子之间形成一个对，并过滤掉问题和上下文句子之间词汇重叠度低的对，将任务转化为句子对分类。任务是判断上下文句子是否包含问题的答案"""
        dt = "qnli"
        question = entry["question"]
        sentence = entry["sentence"]
        answer = "True" if entry["label"] == 1 else "False"
        question = f"""{self.instructs[dt]}. 
 Question: '{question}'
 Sentence: '{sentence}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def rte(self, entry: Dict[str, Any], mode="train"):
        """rte: 文本蕴涵识别(RTE)数据集来自一系列年度文本蕴涵挑战。基准的作者结合了RTE1 (Dagan等人，2006)、RTE2 (Bar Haim等人，2006)、RTE3 (Giampiccolo等人，2007)和RTE5 (Bentivogli等人，2009)的数据。示例是基于新闻和维基百科文本构建的。基准测试的作者将所有数据集转换为两类拆分，其中对于三类数据集，他们将中性和矛盾分解为非蕴涵，以保持一致性。"""
        dt = "rte"
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = "not entailment" if entry["label"] == 1 else "entailment"
        question = f"""{self.instructs[dt]}. 
 Sentence1: '{sentence1}'
 Sentence2: '{sentence2}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def wnli(self, entry: Dict[str, Any], mode="train"):
        """wnli: Winograd图式挑战(Levesque et al.， 2011)是一项阅读理解任务，系统必须阅读一个带有代词的句子，并从选项列表中选择该代词的指代物。这些示例是手动构建的，以挫败简单的统计方法:每个示例都取决于句子中单个单词或短语提供的上下文信息。为了将问题转化为句子对分类，基准的作者通过用每个可能的指称替换歧义代词来构建句子对。任务是预测被替换代词的句子是否包含在原句子中。他们使用了一个小的评估集，该评估集由原始语料库的作者私下共享的来自小说书籍的新示例组成。虽然包含的训练集在两个类之间是平衡的，但测试集在它们之间是不平衡的(65%非蕴涵)。此外，由于数据的特殊性，开发集是对抗性的:假设有时在训练和开发示例之间共享，因此如果模型记住了训练示例，它们将预测错误的实验室"""
        dt = "wnli"
        sentence1 = entry["sentence1"]
        sentence2 = entry["sentence2"]
        answer = "entailment" if entry["label"] == 1 else "not entailment"
        question = f"""{self.instructs[dt]}. 
 Sentence1: '{sentence1}'
 Sentence2: '{sentence2}'
 Answer: """
        return dict(
            question = question,
            answer = answer
        )
    def prefix(self, item, dt):
        """for test

        Parameters
        ----------
        item : _type_
            _description_
        dt : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        idx = {"idx": item["idx"]}
        return f"<idx>{json.dumps(idx)}</idx><l>{dt}</l>{item['answer']}"

    def dataset_process(self, res, df_hf, processor, dt):
        processor = partial(processor, mode="eval")
        ft = df_hf.features
        ft.pop("idx")
        new_df = df_hf.map(processor, remove_columns=list(ft), batched=False)
        new_df = new_df.map(
            lambda x: {
                "question": x["question"],
                "answer": f"{self.prefix(x, dt)}",
            },
            remove_columns=["idx"],
            batched=False,
        )
        res.append(new_df)
    def get_train_dataset(self, size: int = -1):
        ttl_dt = self.params['train_file']
        # ttl_data = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
        res = []
        for dt in ttl_dt:
            train_dataset = datasets.load_dataset(
                self.dataset_name,
                dt,
                cache_dir=self.params.get("cache_dir", None),
                token=self.params.get("token", None),
            )['train']
            train_dataset = train_dataset.shuffle(seed=42)
            max_length = min(len(train_dataset), self.params.get("train_max_size_each_dataset", 1e10))
            train_dataset = train_dataset.select(range(max_length))
            processor = getattr(self, dt, None)
            assert processor is not None, f"the process for {dt} does not exist!"
            processor = partial(processor, mode="train")
            ft = train_dataset.features
            new_df = train_dataset.map(
                processor, remove_columns=list(ft), batched=False
            )
            res.append(new_df)
        res = datasets.concatenate_datasets(res)
        
        # shuffle the dataset
        res = res.shuffle(seed=42)
        if size == -1:
            return res
        else:
            return res.select(range(size))
        
    def get_test_dataset(self, size: int = -1):
        ttl_dt = self.params["test_file"]
        # ttl_data = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
        res = []
        for dt in ttl_dt:
            df_hf = datasets.load_dataset(
                self.dataset_name,
                dt,
                cache_dir=self.params.get("cache_dir", None),
                token=self.params.get("token", None),
            )["test"]
            processor = getattr(self, dt, None)
            if processor:
                self.dataset_process(res, df_hf, processor, dt)
        res = datasets.concatenate_datasets(res)
        if size == -1:
            return res
        else:
            return res.select(range(size))
    def get_eval_dataset(self, size: int = -1):
        ttl_dt = self.params["eval_file"]
        # ttl_data = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"]
        res = []

        for dt in ttl_dt:
            df_hf = datasets.load_dataset(
                self.dataset_name,
                dt,
                cache_dir=self.params.get("cache_dir", None),
                token=self.params.get("token", None),
            )["validation" if dt != "mnli" else "validation_matched"]
            max_length = min(len(df_hf), self.params.get("train_max_size_each_dataset", 1e10))
            eval_dataset = df_hf.select(range(max_length))
            processor = getattr(self, dt, None)
            if processor:
                self.dataset_process(res, eval_dataset, processor, dt)

        res = datasets.concatenate_datasets(res)
        if size > 0:
            res = res.shuffle(seed=42)
        if size == -1:
            return res
        else:
            return res.select(range(size))
    def collate_fn(
        self,
        batch: List[Dict[str, Any]],
        tokenizer=None,
        max_len: int = 1024,
        batch_pad_truncate: bool = True,
        ignore_q: bool = True,
        mode: str = "train",
    ):
        """used for dataset preprocess

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            will get a batch data from sampler
        tokenizer : _type_, optional
            common tokenizer, by default None
        max_len : int, optional
            the max length of dataset, if exceed, truncate it, by default 1024
        batch_pad_truncate : bool, optional
            truncate all the data based on the longest data in a batch, by default True
        ignore_q : bool, optional
            whether use loss mask or not, by default True
        mode : str, optional
            train or test, by default "train"

        Returns
        -------
        _type_
            _description_
        """
        inputs, attention_mask, labels, raw_labels, ground_truths = (
            [],
            [],
            [],
            [],
            [],
        )
        current_max_len = 0
        current_max_len_truth = 0

        for b in batch:
            # if 'origin' not in b:
            #     logger.info('no origin found for this data point...')
            #     logger.info(b)
            #     origin = 'record'
            # else: origin = b.pop('origin')
            # origin_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(origin))
            question, answer = b["question"], b["answer"]

            current_input_tokens = [tokenizer.bos_token] + tokenizer.tokenize(
                question
            )

            current_label_ids = [IGNORE_INDEX] * len(current_input_tokens)
            raw_label_ids = tokenizer.convert_tokens_to_ids(
                current_input_tokens
            )
            # for ground truth
            gd_truth_tokens = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(answer)
            )
            ground_truths.append(gd_truth_tokens)

            # now add the label tokens
            if mode == "train":
                output_eos_tokens = tokenizer.tokenize(answer) + [
                    tokenizer.eos_token
                ]
                current_input_tokens.extend(output_eos_tokens)
                output_ids = tokenizer.convert_tokens_to_ids(output_eos_tokens)
                current_label_ids.extend(output_ids)
                raw_label_ids.extend(output_ids)

            # record the max length
            current_max_len = max(current_max_len, len(current_input_tokens))
            current_max_len_truth = max(
                current_max_len_truth, len(gd_truth_tokens)
            )

            # append to target elements
            inputs.append(tokenizer.convert_tokens_to_ids(current_input_tokens))
            labels.append(current_label_ids)
            raw_labels.append(raw_label_ids)
            attention_mask.append([1] * len(current_input_tokens))

        if batch_pad_truncate:
            padding_length = min(current_max_len, max_len)
        else:
            padding_length = max_len

        if mode == "train":
            for c in range(len(inputs)):
                if len(inputs[c]) < padding_length:
                    inputs[c] += [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    )
                    labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(labels[c])
                    )
                    raw_labels[c] += [IGNORE_INDEX] * (
                        padding_length - len(raw_labels[c])
                    )
                    attention_mask[c] += [0] * (
                        padding_length - len(attention_mask[c])
                    )
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    labels[c] = labels[c][-padding_length:]
                    raw_labels[c] = raw_labels[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        else:
            for c in range(len(inputs)):
                if len(ground_truths[c]) < current_max_len_truth:
                    ground_truths[c].extend(
                        [tokenizer.pad_token_id]
                        * (current_max_len_truth - len(ground_truths[c]))
                    )
                if len(inputs[c]) < padding_length:
                    inputs[c] = [tokenizer.pad_token_id] * (
                        padding_length - len(inputs[c])
                    ) + inputs[
                        c
                    ]  ## padding to the left for generation
                    attention_mask[c] = [0] * (
                        padding_length - len(attention_mask[c])
                    ) + attention_mask[c]
                else:
                    inputs[c] = inputs[c][-padding_length:]
                    attention_mask[c] = attention_mask[c][-padding_length:]
        if not ignore_q:
            labels = raw_labels

        if mode == "train":
            return {
                "input_ids": torch.tensor(inputs),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "raw_labels": torch.tensor(raw_labels),
            }
        return {
            "input_ids": torch.tensor(inputs),
            "attention_mask": torch.tensor(attention_mask),
            "ground_truth_labels": torch.tensor(ground_truths),
        }

    def metric(self, p):
        return super().metric(p)
    
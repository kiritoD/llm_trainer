# -*- coding: utf-8 -*-
import json
import re
import os
import string
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import evaluate
import jsonlines
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score

from .super_gelu_process import SuperGeluResult

warnings.filterwarnings("ignore")


def boolq_acc(path):
    jr = jsonlines.Reader(open(path, "r"))

    pred, label, probs = [], [], []
    for line in jr:
        if line:
            x, y = (
                line["answer"].lower().strip(),
                line["ground_truth"].lower().strip(),
            )
            if y == "true":
                label.append(1)
            else:
                label.append(0)

            if ("true" in x) and ("false" in x):
                pred.append(-1)
            elif ("true" in x) and ("false" not in x):
                pred.append(1)
            elif ("true" not in x) and ("false" in x):
                pred.append(0)
            else:
                pred.append(-1)

    assert len(pred) == len(label)
    # print("accuracy:", accuracy_score(label, pred))
    return {"accuracy": accuracy_score(label, pred)}


def siqa_acc(path):
    jr = jsonlines.Reader(open(path, "r"))

    pred, label, probs = [], [], []
    for line in jr:
        if line:
            x, y = (
                line["answer"].lower().strip(),
                line["ground_truth"].lower().strip(),
            )
            if y == "a":
                label.append(1)
            elif y == "b":
                label.append(2)
            elif y == "c":
                label.append(3)

            if x == "a":
                pred.append(1)
            elif x == "b":
                pred.append(2)
            elif x == "c":
                pred.append(3)
            else:
                pred.append(4)

    assert len(pred) == len(label)
    # print("accuracy:", accuracy_score(label, pred))
    return {"accuracy": accuracy_score(label, pred)}


def record_metrics(path):
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        # 对相同字符的数量进行统计（粒度到每个字母）
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        # 只需要一个prediction，只有一个prediction和多个答案求相同字符数量，取其中最大的一个数
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    f1 = exact_match = total = 0
    # correct_ids = []
    jr = jsonlines.Reader(open(path, "r"))
    for line in jr:
        if line:
            pred_, label_list = line["answer"], line[
                "ground_truth"
            ].lower().strip(", ")
            _exact_match = metric_max_over_ground_truths(
                exact_match_score, pred_, label_list
            )
            # if int(_exact_match) == 1:
            #     correct_ids.append(total)
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, pred_, label_list)
        total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print(
        "Total queries: {} | Exact_match: {} | F1: {}".format(
            total, exact_match, f1
        )
    )

    return {"exact_match": exact_match, "f1": f1}


def cnn_dailymail_metrics(path):
    rouge = evaluate.load("rouge")
    jr = jsonlines.Reader(open(path, "r"))
    predictions = []
    references = []
    for line in jr:
        predictions.append(line["answer"])
        references.append(line["ground_truth"])

    metrics = rouge.compute(predictions=predictions, references=references)
    return metrics


def samsum_metric(path):
    rouge = evaluate.load("rouge")
    jr = jsonlines.Reader(open(path, "r"))
    predictions = []
    references = []
    for line in jr:
        predictions.append(line["answer"])
        references.append(line["ground_truth"])

    metrics = rouge.compute(predictions=predictions, references=references)
    return metrics


def xsum_metrics(path):
    rouge = evaluate.load("rouge")
    jr = jsonlines.Reader(open(path, "r"))
    predictions = []
    references = []
    for line in jr:
        predictions.append(line["answer"])
        references.append(line["ground_truth"])

    metrics = rouge.compute(predictions=predictions, references=references)
    return metrics


def gsm8k_metrics(path):
    rouge = evaluate.load("rouge")

    jr = jsonlines.Reader(open(path, "r"))
    accuracy = 0
    predictions = []
    references = []
    predictions_str = []
    references_str = []

    for line in jr:
        predictions_str.append(line["ground_truth"])
        references_str.append(line["answer"])
        references.append(int(re.findall(r"\d+", line["ground_truth"])[-1]))
        number_arr = re.findall(r"\d+", line["answer"].split(".")[0])
        if len(number_arr) > 0:
            predictions.append(int(number_arr[-1]))
        else:
            predictions.append(-1e-6)
    for index, value in enumerate(predictions):
        if value == references[index]:
            accuracy += 1
            print(value, references[index])
    metrics = rouge.compute(
        predictions=predictions_str, references=references_str
    )
    metrics.update({"acc": accuracy * 100 / len(references)})
    return metrics


def super_glue_metric(path, if_only_zip=False):
    """
    if `if_only_zip` == False, evaluate in validation dataset, else, evaluate in test dataset
    """
    if not if_only_zip:
        search_re = re.compile(r"<l>(.*)</l>")
        search_re_idx = re.compile(r"<idx>(.*)</idx>")
        jr = jsonlines.Reader(open(path, "r"))

        metrics = {}
        pred, label, idx = {}, {}, {}
        for line in jr:
            if line:
                x, y = (
                    line["answer"].lower().strip(),
                    line["ground_truth"].lower().strip(),
                )
            dt_re = search_re.search(y)
            if dt_re:
                dt = dt_re.group(1)
                idx_str = search_re_idx.search(y).group(1)
                idx_item = json.loads(idx_str)["idx"]
                label_pre = y[dt_re.span(0)[-1] :]
                if dt not in pred:
                    pred[dt] = []
                    label[dt] = []
                    idx[dt] = []
                pred[dt].append(x)
                label[dt].append(label_pre)
                idx[dt].append(idx_item)
        for dt in pred:
            metric = SuperGlueMetricLoader.metric_handler(
                dt, pred[dt], label[dt], idx=idx[dt]
            )
            metrics[dt] = metric

        return metrics
    else:
        s = SuperGeluResult()
        s.synthesis(
            path,
            os.path.join(os.path.dirname(path), "super_glue_test"),
        )
        return {}


class SuperGlueMetricLoader:
    @classmethod
    def metric_handler(cls, dt, pred, label, idx):
        metric_cal = evaluate.load("super_glue", dt)
        collator = getattr(cls, dt, None)
        if collator:
            p, ref = collator(pred, label, idx=idx)
            metric = metric_cal.compute(predictions=p, references=ref)
        return metric

    @classmethod
    def boolq(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, label)
        )
        return pred, label

    @classmethod
    def cb(cls, pred, label, **kwargs):
        pred = list(
            map(
                lambda x: 0
                if x == "entailment"
                else 1
                if x == "contradiction"
                else 2
                if x == "neutral"
                else 3,
                pred,
            )
        )
        label = list(
            map(
                lambda x: 0
                if x == "entailment"
                else 1
                if x == "contradiction"
                else 2
                if x == "neutral"
                else 3,
                label,
            )
        )

        return pred, label

    @classmethod
    def copa(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 0 if x == "1" else 1 if x == "2" else 2, pred)
        )
        label = list(
            map(lambda x: 0 if x == "1" else 1 if x == "2" else 2, label)
        )
        return pred, label

    @classmethod
    def multirc(cls, pred, label, idx):
        label = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, label)
        )

        pred_process = []
        for i in range(len(pred)):
            pred_process.append(
                {
                    "idx": idx[i],
                    # 'prediction': 1 if pred[i] == 'true' else 0 if pred[i] == 'false' else 2
                    "prediction": 1 if pred[i] == "true" else 0,
                }
            )
        return pred_process, label

    @classmethod
    def record(cls, pred, label, idx):
        pred_process, label_process = [], []
        for i in range(len(pred)):
            pred_process.append({"idx": idx[i], "prediction_text": pred[i]})
            label_process.append(
                {"idx": idx[i], "answers": label[i].split(", ")}
            )
        return pred_process, label_process

    @classmethod
    def rte(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 0 if x == "yes" else 1 if x == "no" else 2, pred)
        )
        label = list(
            map(lambda x: 0 if x == "yes" else 1 if x == "no" else 2, label)
        )

        return pred, label

    @classmethod
    def wic(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, label)
        )

        return pred, label

    @classmethod
    def wsc(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0 if x == "false" else 2, label)
        )

        return pred, label


def glue_metric(path, if_only_zip=False):
    if not if_only_zip:
        search_re = re.compile(r"<l>(.*)</l>")
        search_re_idx = re.compile(r"<idx>(.*)</idx>")
        jr = jsonlines.Reader(open(path, "r"))

        metrics = {}
        pred, label, idx = {}, {}, {}
        for line in jr:
            if line:
                x, y = (
                    line["answer"].lower().strip(),
                    line["ground_truth"].lower().strip(),
                )
            dt_re = search_re.search(y)
            if dt_re:
                dt = dt_re.group(1)
                idx_str = search_re_idx.search(y).group(1)
                idx_item = json.loads(idx_str)["idx"]
                label_pre = y[dt_re.span(0)[-1] :]
                if dt not in pred:
                    pred[dt] = []
                    label[dt] = []
                    idx[dt] = []
                pred[dt].append(x)
                label[dt].append(label_pre)
                idx[dt].append(idx_item)
        for dt in pred:
            metric = GlueMetricLoader.metric_handler(
                dt, pred[dt], label[dt], idx=idx[dt]
            )
            metrics[dt] = metric

        return metrics
    else:
        s = SuperGeluResult()
        s.synthesis(
            path,
            os.path.join(os.path.dirname(path), "glue_test"),
        )
        return {}
    
class GlueMetricLoader:
    @classmethod
    def metric_handler(cls, dt, pred, label, idx):
        metric_cal = evaluate.load("glue", dt)
        collator = getattr(cls, dt, None)
        if collator:
            p, ref = collator(pred, label, idx=idx)
            metric = metric_cal.compute(predictions=p, references=ref)
        return metric

    @classmethod
    def cola(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0, label)
        )
        return pred, label

    @classmethod
    def sst2(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0, label)
        )
        return pred, label

    @classmethod
    def mrpc(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0, label)
        )
        return pred, label

    @classmethod
    def stsb(cls, pred, label, **kwargs):
        def _convert_to_float(text):
            try:
                return float(text)
            except ValueError:
                return float(0.0)
        pred = list(
            map(_convert_to_float, pred)
        )
        label = list(
            map(_convert_to_float, label)
        )
        return pred, label

    @classmethod
    def qqp(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0, label)
        )
        return pred, label

    @classmethod
    def mnli(cls, pred, label, **kwargs):
        pred = list(
            map(
                lambda x: 0
                if x == "entailment"
                else 2
                if x == "contradiction"
                else 1,
                pred,
            )
        )
        label = list(
            map(
                lambda x: 0
                if x == "entailment"
                else 2
                if x == "contradiction"
                else 1,
                label,
            )
        )

        return pred, label

    @classmethod
    def qnli(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "true" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "true" else 0, label)
        )
        return pred, label

    @classmethod
    def rte(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "not entailment" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "not entailment" else 0, label)
        )
        return pred, label
    @classmethod
    def wnli(cls, pred, label, **kwargs):
        pred = list(
            map(lambda x: 1 if x == "entailment" else 0, pred)
        )
        label = list(
            map(lambda x: 1 if x == "entailment" else 0, label)
        )
        return pred, label

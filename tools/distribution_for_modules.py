import os
import time
from tkinter import font
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open
import re
import pandas as pd

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


# where the method in class PeftMerge
PEFT_TYPE_MAPPING_PROCESS = {"lora": "lora_mul", "vera": "vera_mul"}
PEFT_TYPE_MAPPING_KEY = {
    "lora": ["lora_A", "lora_B"],
    "vera": ["vera_lambda_b", "vera_lambda_d"],
}
TARGET_MODULES = {
    "roberta-base": [
        "key",
        "dense",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
        "query",
        "value",
    ],
    "llama": [
        "v_proj",
        # "down_proj",
        # "up_proj",
        "q_proj",
        "gate_proj",
        # "lm_head",
        "k_proj",
        "o_proj",
    ],
}

LABEL_MAPPING = {
    "roberta-base": {
        "query": "q",
        "key": "k",
        "value": "v",
        "attention.output.dense": "o",
        "intermediate.dense": "up",
        "output.dense": "down",
    },
    "llama": {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "gate_proj": "g",
        # "up_proj": "up",
        # "down_proj": "down",
    },
}

MODEL_AB_NAMES = ["vera"]
MODEL_AB_MAPPING = {"vera": ["vera_A", "vera_B"]}
# reg group
NUMBER_RE = re.compile(r"\d+")


def reg_pattern(model_type):
    target_moudels = TARGET_MODULES[model_type]
    module_re = re.compile(r"|".join(target_moudels))
    return module_re


class Visualization:
    def __init__(
        self,
        output_dir: str = "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/ditributions",
    ):
        self.output_dir = output_dir

    def hist(self, data, model_name, dt, alpha=0.8):
        self.output_dir = os.path.join(self.output_dir, "hist")
        os.makedirs(self.output_dir, exist_ok=True)
        # 直方图绘制
        for layer_number, weights in data.items():
            title = f"Weight Analysis of layer_{layer_number} in {model_name} for {dt}"

            if isinstance(data, dict):
                for key, value in weights.items():
                    target_value = value["merge_result"].view(-1).numpy()
                    plt.hist(target_value, bins=30, alpha=alpha, label=key)
            else:
                # 绘制直方图
                data = weights.view(-1).numpy()
                plt.hist(data, bins=30, color="skyblue", alpha=alpha)

            # 设置图表属性
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()

            # 显示图表

            plt.savefig(
                f"{self.output_dir}/hist_{model_name}_{dt}_layer_{layer_number}.jpg"
            )
            # plt.show()
            plt.clf()

    def violin(self, data, model_name, dt, alpha=0.8):
        start = time.time()
        self.output_dir = os.path.join(self.output_dir, "violin")
        os.makedirs(self.output_dir, exist_ok=True)
        data = dict(sorted(data.items(), key=lambda x: int(x[0])))
        # 折线图绘制
        length = len(data)
        nrow = int(np.sqrt(length))
        ncols = nrow + 1 if (nrow + 1) * nrow >= length else nrow + 2
        fig, axs = plt.subplots(nrows=nrow, ncols=ncols, figsize=(ncols * 9, nrow * 4))
        x = 0
        y = 0
        title = f"Weight Analysis of {model_name} for {dt}"
        index = 0
        for layer_number, weights in data.items():
            x = index // ncols
            y = index % ncols
            keys = []
            values = []
            for key, value in weights.items():
                target_value = value["merge_result"].view(-1).numpy()
                # target_value = target_value[:1000]
                values.append(target_value)
                keys.append(key)
            axs[x][y].violinplot(values, showmeans=True, showmedians=True)
            # axs[x][y].boxplot(values, showfliers=False, notch=True)
            axs[x][y].set_title(f"layer_{layer_number}")
            index += 1

        # 设置图表属性
        plt.title(title)
        # plt.xlabel("module")
        # plt.ylabel("value")
        plt.legend()

        plt.savefig(f"{self.output_dir}/violin_{model_name}_{dt}.jpg")
        # plt.show()
        plt.clf()
        print(f"{time.time() - start:.3f}s")

    def boxplot(self, data, model_name, dt, alpha=0.8):
        method_name = "boxplot"
        start = time.time()
        data_new = {}
        for layer_number, weights in data.items():
            if weights[list(weights.keys())[0]].get("plot", False) == True:
                data_new[layer_number] = weights
        self.output_dir = os.path.join(self.output_dir, method_name)
        os.makedirs(self.output_dir, exist_ok=True)
        data_new = dict(sorted(data_new.items(), key=lambda x: int(x[0])))
        # 折线图绘制
        length = len(data_new)
        nrow = int(np.sqrt(length))
        ncols = nrow + 1 if (nrow + 1) * nrow >= length else nrow + 2
        fig, axs = plt.subplots(
            nrows=nrow,
            ncols=ncols,
            figsize=(ncols * 9, nrow * 4),
            sharey=True,
            sharex=True,
        )
        x = 0
        y = 0
        title = f"Weight Analysis of {model_name} for {dt}"
        index = 0
        for layer_number, weights in data_new.items():
            x = index // ncols
            y = index % ncols
            keys = []
            values = []
            for key, value in LABEL_MAPPING[model_name].items():
                target_value = weights[key]["merge_result"].float().view(-1).numpy()
                # target_value = target_value[:1000]
                values.append(target_value)
                keys.append(value)
            # axs[x][y].violinplot(values, showmeans=True, showmedians=True)
            # print(keys)
            colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
            bplot = axs[x][y].boxplot(
                values,
                showfliers=False,
                patch_artist=True,
                whis=(5, 95),
                tick_labels=keys,
            )
            axs[x][y].tick_params(axis="x", labelsize=18, width=1)
            axs[x][y].set_title(f"Layer {layer_number}", fontsize=18)
            # fill with colors
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)
            index += 1

        # 设置图表属性
        plt.suptitle(title, fontsize=28)
        # plt.xlabel("module")
        # plt.ylabel("value")
        # plt.ylim(-0.001, 0.001)
        # plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.legend()

        plt.savefig(f"{self.output_dir}/{method_name}_{model_name}_{dt}.jpg")
        # plt.show()
        plt.clf()
        print(f"{time.time() - start:.3f}s")

    def boxplot_layer(self, data, model_name, dt, alpha=0.8):
        method_name = "boxplot_layer"
        start = time.time()
        data_new = {}
        for layer_number, weights in data.items():
            if weights[list(weights.keys())[0]].get("plot", False) == True:
                data_new[layer_number] = weights
        self.output_dir = os.path.join(self.output_dir, method_name)
        os.makedirs(self.output_dir, exist_ok=True)
        data_new = dict(sorted(data_new.items(), key=lambda x: int(x[0])))
        # 折线图绘制
        data_layer = {}
        for layer_number, weights in data_new.items():
            for module, value in weights.items():
                if module not in data_layer:
                    data_layer[module] = {layer_number: value}
                else:
                    data_layer[module][layer_number] = value
        length = len(data_layer)
        nrow = int(np.sqrt(length)) + 1
        ncols = nrow - 1 if (nrow - 1) * nrow >= length else nrow
        fig, axs = plt.subplots(
            nrows=nrow,
            ncols=ncols,
            figsize=(nrow * 9, ncols * 6),
            sharey=True,
            sharex=True,
        )
        x = 0
        y = 0
        title = f"Weight Analysis of {model_name} for {dt}"
        index = 0
        colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
        for key, value in LABEL_MAPPING[model_name].items():
            x = index // ncols
            y = index % ncols
            keys = []
            values = []
            for layer_number, weights in data_layer[key].items():
                target_value = weights["merge_result"].float().view(-1).numpy()
                # target_value = target_value[:1000]
                values.append(target_value)
                keys.append(layer_number)
            colors_ = [colors[index]] * len(keys)
            bplot = axs[x][y].boxplot(
                values,
                showfliers=False,
                patch_artist=True,
                whis=(5, 95),
                tick_labels=keys,
            )
            axs[x][y].tick_params(axis="x", labelsize=18, width=1)
            axs[x][y].set_title(f"module {value}", fontsize=18)
            # fill with colors
            for patch, color in zip(bplot["boxes"], colors_):
                patch.set_facecolor(color)
            index += 1

        # 设置图表属性
        plt.suptitle(title, fontsize=28)
        # plt.xlabel("layer index", fontsize=24)
        # plt.ylabel("value")
        # plt.ylim(-0.001, 0.001)
        # plt.xticks(fontsize=18)
        plt.tight_layout()
        plt.legend()

        plt.savefig(f"{self.output_dir}/{method_name}_{model_name}_{dt}.jpg")
        # plt.show()
        plt.clf()
        print(f"{time.time() - start:.3f}s")

    def line(self, data, model_name, dt, alpha=0.8):
        method_name = "line"
        start = time.time()
        self.output_dir = os.path.join(self.output_dir, method_name)
        os.makedirs(self.output_dir, exist_ok=True)
        data = dict(sorted(data.items(), key=lambda x: int(x[0])))
        # 折线图绘制
        length = len(data)
        nrow = int(np.sqrt(length))
        ncols = nrow + 1 if (nrow + 1) * nrow >= length else nrow + 2
        fig, axs = plt.subplots(nrows=nrow, ncols=ncols, figsize=(ncols * 9, nrow * 4))
        x = 0
        y = 0
        title = f"Weight Analysis of {model_name} for {dt}"
        index = 0
        for layer_number, weights in data.items():
            x = index // ncols
            y = index % ncols
            keys = []
            values = []
        for key, value in weights.items():
            target_value = value["merge_result"].view(-1).numpy()
            # target_value = target_value[:1000]
            values.append(target_value)
            keys.append(key)
        # axs[x][y].violinplot(values, showmeans=True, showmedians=True)
        colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
        bplot = axs[x][y].boxplot(values, showfliers=False, patch_artist=True)
        axs[x][y].set_title(f"layer_{layer_number}")
        # fill with colors
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
        index += 1

        # 设置图表属性
        plt.title(title)
        # plt.xlabel("module")
        # plt.ylabel("value")
        plt.legend()

        plt.savefig(f"{self.output_dir}/{method_name}_{model_name}_{dt}.jpg")
        # plt.show()
        plt.clf()
        print(f"{time.time() - start:.3f}s")

    def density(self, data, model_name, dt):
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        method_name = "density"
        start = time.time()
        self.output_dir = os.path.join(self.output_dir, method_name)
        os.makedirs(self.output_dir, exist_ok=True)
        data = dict(sorted(data.items(), key=lambda x: int(x[0])))

        # Create the data
        for layer_number, weights in data.items():
            key_label = []
            keys = []
            values = []
            for key, value in weights.items():
                target_value = value["merge_result"].view(-1)
                target_value = torch.exp(target_value)
                target_value = target_value.numpy()
                # target_value = target_value[:1000]
                values.extend(list(target_value))
                keys.extend([key] * len(target_value))
                key_label.append(key)
            df = pd.DataFrame(dict(g=keys, x=values))

            # rs = np.random.RandomState(1979)
            # x = rs.randn(50000)
            # g = np.tile(list("ABCDEFGHIJ"), 5000)
            # df = pd.DataFrame(dict(x=x, g=g))
            # m = df.g.map(ord)
            # df["x"] += m

            # Initialize the FacetGrid object
            pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
            pal = sns.color_palette("Set2", 12)
            g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=0.5, palette=pal)

            # Draw the densities in a few steps
            g.map(
                sns.kdeplot,
                "x",
                bw_adjust=0.5,
                clip_on=False,
                fill=True,
                alpha=1,
                linewidth=1.5,
            )
            g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=0.5)

            # passing color=None to refline() uses the hue mapping
            g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x, color, label):
                ax = plt.gca()
                ax.text(
                    0,
                    0.2,
                    label,
                    fontweight="bold",
                    color=color,
                    ha="left",
                    va="center",
                    transform=ax.transAxes,
                )

            g.map(label, "x")

            # Set the subplots to overlap
            g.figure.subplots_adjust(hspace=-0.25)

            # Remove axes details that don't play well with overlap
            g.set_titles("")
            g.set(yticks=[], xlabel="Value", ylabel="")
            g.despine(bottom=True, left=True)
            plt.suptitle("Netflix Originals - IMDB Scores by Language", y=0.98)
            plt.savefig(f"{self.output_dir}/{method_name}_{model_name}_{dt}.jpg")
            # plt.show()
            plt.clf()
            print(f"{time.time() - start:.3f}s")
            quit()


class PeftMerge:
    # 参数名字必须和PEFT_TYPE_MAPPING_KEY以及MODEL_AB_MAPPING里的一一对应
    @classmethod
    def lora_mul(cls, lora_A, lora_B):
        return lora_B @ lora_A

    @classmethod
    def vera_mul(cls, vera_lambda_b, vera_lambda_d, vera_B, vera_A):
        sliced_A = vera_A[:, : vera_lambda_b.shape[0]]
        sliced_B = vera_B[: vera_lambda_b.shape[0], :]
        result = (vera_lambda_b.unsqueeze(-1) * sliced_B) @ (
            vera_lambda_d.unsqueeze(-1) * sliced_A
        )
        return result


def weight_load(path: str):
    tensors = {}
    if path.endswith("safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    elif path.endswith("bin"):
        tensors = torch.load(path, map_location="cpu")

    return tensors


def weight_parse(model_weights: dict, model_type: str, peft_type: str):
    # 对参数进行逐层逐模块解析
    module_re = reg_pattern(model_type)
    model = {}
    if peft_type in MODEL_AB_NAMES:
        for key in MODEL_AB_MAPPING[peft_type]:
            key_ = f"base_model.{key}"
            model[key_] = model_weights[key_]

    for key, value in model_weights.items():
        value_item = {key: value}
        # 找到对应的层编号
        numbers = NUMBER_RE.findall(key)
        # 找到对应的模块名字
        target_modules = module_re.findall(key)
        # 赋值对应的参数
        if len(numbers) > 0 and len(target_modules) > 0:
            layer_number = numbers[0]
            target_module = target_modules[0]
            if layer_number not in model:
                model[f"{layer_number}"] = {target_module: value_item}
            else:
                if target_module not in model[f"{layer_number}"]:
                    model[f"{layer_number}"][target_module] = value_item
                else:
                    model[f"{layer_number}"][target_module].update(value_item)
    return model


def weight_merge(model_weights_parse: dict, peft_type: str, model_type: str):
    # 根据特定的merge策略进行参数合并
    merge_method = PEFT_TYPE_MAPPING_PROCESS[peft_type]
    merge_key = PEFT_TYPE_MAPPING_KEY[peft_type]
    args = {_: None for _ in merge_key}
    if peft_type in MODEL_AB_NAMES:
        for key in MODEL_AB_MAPPING[peft_type]:
            key_ = f"base_model.{key}"
            args[key] = model_weights_parse.pop(key_, None)

    for layer_number, layer in model_weights_parse.items():
        # llama的参数量太大，当前设备不足以全部处理
        if int(layer_number) % 4 != 0 and model_type == "llama":
            continue
        for target_module, module_weights in layer.items():
            for key in args:
                for weight_name, weight_value in module_weights.items():
                    if key.lower() in weight_name.lower():
                        args[key] = weight_value
            # 调用对应的方法进行参数合并
            model_weights_parse[layer_number][target_module]["merge_result"] = getattr(
                PeftMerge, merge_method
            )(**args)
            model_weights_parse[layer_number][target_module]["plot"] = True
    return model_weights_parse


def main(
    path: str,
    model_type: str,
    peft_type: str,
    output_dir: str,
    dataset_name: str,
    method: str,
):
    model_weights = weight_load(path)
    model_weights_parse = weight_parse(model_weights, model_type, peft_type)
    model_weights_merge = weight_merge(model_weights_parse, peft_type, model_type)
    visualization = Visualization(output_dir)
    # visualization.hist(model_weights_merge, model_type, dataset_name)
    method_ = getattr(visualization, method, "None")
    if method_ != "None":
        method_(model_weights_merge, model_type, dataset_name)


if __name__ == "__main__":
    roberta_weight_lora_path_mapping = {
        "sst2": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/sst2/LORA_A/step_1565/adapter_model.safetensors",
        "mnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/mnli/LORA_A/step_1565/adapter_model.safetensors",
        "cola": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/cola/LORA_A/step_5360/adapter_model.safetensors",
        "mrpc": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/mrpc/LORA_A/step_2300/adapter_model.safetensors",
        "qnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/qnli/LORA_A/step_6260/adapter_model.safetensors",
        "qqp": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/qqp/LORA_A/step_6260/adapter_model.safetensors",
        "rte": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/rte/LORA_A/step_1560/adapter_model.safetensors",
        "stsb": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/stsb/LORA_A/step_3600/adapter_model.safetensors",
        "wnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/lora/wnli/LORA_A/step_400/adapter_model.safetensors",
    }
    roberta_weight_vera_path_mapping = {
        "sst2": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/sst2/VERA/step_1565/adapter_model.safetensors",
        "mnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/mnli/VERA/step_1565/adapter_model.safetensors",
        "cola": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/cola/VERA/step_8040/adapter_model.safetensors",
        "mrpc": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/mrpc/VERA/step_2300/adapter_model.safetensors",
        "qnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/qnli/VERA/step_6260/adapter_model.safetensors",
        "qqp": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/qnli/VERA/step_6260/adapter_model.safetensors",
        "rte": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/qnli/VERA/step_6260/adapter_model.safetensors",
        "stsb": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/qnli/VERA/step_6260/adapter_model.safetensors",
        "wnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/hssa/distribution_analysis/roberta/vera/qnli/VERA/step_6260/adapter_model.safetensors",
    }
    llama2_weight_lora_path_mapping = {
        "sst2": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/sst2/LORA/step_780/adapter_model.bin",
        # "mnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/mnli/LORA/step_780/adapter_model.bin",
        # "cola": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/cola/LORA/step_670/adapter_model.bin",
        # "mrpc": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/mrpc/LORA/step_285/adapter_model.bin",
        # "qnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/qnli/LORA/step_780/adapter_model.bin",
        # "qqp": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/qqp/LORA/step_780/adapter_model.bin",
        # "rte": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/rte/LORA/step_195/adapter_model.bin",
        # "stsb": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/stsb/LORA/step_450/adapter_model.bin",
        # "wnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/wnli/LORA/step_50/adapter_model.bin",
        # "all": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/all/LORA/step_4775/adapter_model.bin",
    }
    target_mapping: dict = llama2_weight_lora_path_mapping
    peft_type = "lora"
    method = "boxplot_layer"
    model_type = "llama"
    # model_type = "llama"
    for dataset_name, path in target_mapping.items():
        path = path
        # output_dir = os.path.join(os.path.dirname(path), "distribution_analysis")
        output_dir = os.path.join(
            f"/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/ditributions/{model_type}",
        )
        dataset_name = dataset_name
        # print(path, dataset_name)
        print(f"start to analysis model `{path}`")
        main(path, model_type, peft_type, output_dir, dataset_name, method)

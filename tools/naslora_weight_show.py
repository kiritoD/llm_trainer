import json
import os
import sys

sys.path.append("/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm")
import re

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from src.utils.auxiliary import load_yaml

search_rank_dict = {
    "lora_r2": 2,
    "lora_gelu_r2": 2,
    "lora_d_parts_r2": 2,
    "lora_r4": 4,
    "lora_gelu_r4": 4,
    "lora_d_parts_r4": 4,
    "lora_r8": 8,
    "lora_gelu_r8": 8,
    "lora_d_parts_r8": 8,
}
# search_rank_dict = {
#     'lora_r1': 1,
#     'lora_r2': 2,
#     'lora_r3': 3,
#     'lora_r4': 4,
#     'lora_r5': 5,
#     'lora_r6': 6,
#     'lora_r7': 7,
#     'lora_r8': 8,
#     'lora_r9': 9,
#     'lora_r10': 10,
#     'lora_r11': 11,
#     'lora_r12': 12,
# }

search_type_dict = {
    "lora_r2": 1,
    "lora_r4": 1,
    "lora_r8": 1,
    "lora_gelu_r4": 2,
    "lora_gelu_r2": 2,
    "lora_gelu_r8": 2,
    "lora_d_parts_r2": 3,
    "lora_d_parts_r4": 3,
    "lora_d_parts_r8": 3,
}
# search_type_dict = {
#     'lora_r1': 1,
#     'lora_r2': 1,
#     'lora_r3': 1,
#     'lora_r4': 1,
#     'lora_r5': 1,
#     'lora_r6': 1,
#     'lora_r7': 1,
#     'lora_r8': 1,
#     'lora_r9': 1,
#     'lora_r10': 1,
#     'lora_r11': 1,
#     'lora_r12': 1,
# }
ffn_type_dict = {
    "q_proj": 1,
    "k_proj": 2,
    "v_proj": 3,
    "o_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "lm_head": 8,
}


def get_points(path, search_space_path):
    search_space = load_yaml(search_space_path)
    search_space_keys = list(search_space.keys())

    with open(path, "r") as f:
        weights_data = json.load(f)
    choices = []
    for key, value in weights_data.items():
        choices_ = [key]
        for index, choice in enumerate(value):
            if choice == 1:
                # choices_.append(index)
                choices_.append(search_rank_dict[search_space_keys[index]])
                choices_.append(search_type_dict[search_space_keys[index]])
        choices.append(choices_)

    number_re = re.compile("\d")
    data_post = list(
        map(
            lambda x: [
                "".join(number_re.findall(x[0])),
                x[0].split(".")[-1],
                x[1],
                x[2],
            ],
            choices,
        )
    )
    ffn_type = set([_[1] for _ in data_post])
    # print(data_post)

    data_post_new = list(
        map(
            lambda x: [
                int(x[0]) if x[0] != "" else "",
                ffn_type_dict[x[1]],
                x[2],
                x[3],
            ],
            data_post,
        )
    )
    data_post_new[-1][0] = data_post_new[-2][0] + 1
    print(ffn_type_dict)
    heat_points = np.array(data_post_new)
    return heat_points


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = None
    if cbarlabel != "":
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, va="top")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(
        np.arange(data.shape[1]),
        labels=col_labels,
    )
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heat(heat_points):
    plt.rcParams["font.family"] = "Times New Roman"
    color_list = [(0.282, 0.792, 0.894), (0, 0.588, 0.78), (0, 0.243, 0.541)]
    cmap1 = colors.ListedColormap(color_list)
    # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15, 6))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(24, 4))
    # ax1.set_title("Visualization of the rank choices about LoRA-like modules")
    # ax1.set_xlabel("transfomer block index")
    # ax1.set_ylabel("different rank of LoRA module")
    ax1.set_xticks(range(0, heat_points[-1][0] + 1))
    ax1.set_yticks(range(1, len(ffn_type_dict)))
    # pos_1 = ax1.scatter(heat_points[:, 0], heat_points[:, 1], c=heat_points[:, 2], s=80, cmap=cmap2)
    x_1, y_1, z_1 = (
        heat_points[:, 0][:-1].reshape(32, 7)[:, [0, 1, 2, 3, 4, 6]],
        heat_points[:, 1][:-1].reshape(32, 7)[:, [0, 1, 2, 3, 4, 6]],
        heat_points[:, 2][:-1].reshape(32, 7).T[::-1, :],
    )

    im_1, cbar_1 = heatmap(
        z_1[:4, :],
        [
            # "q_proj",
            # "k_proj",
            # "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax1,
        cmap=cmap1,
        # cbar_kw={"ticks": [3, 5, 7], "labels": ["2", "4", "8"]},
        cbarlabel="Rank Setting",
    )
    if cbar_1:
        cbar_1.set_ticks(ticks=[3, 5, 7], labels=["2", "4", "8"])

    # ax2.set_title("Visualization of type choices about the LoRA-like modules")
    # ax2.set_xlabel("transfomer block index")
    # ax2.set_ylabel("different linear module")
    ax2.set_xticks(range(0, heat_points[-1][0] + 1))
    ax2.set_yticks(range(1, len(ffn_type_dict)))
    # print(heat_points[:, 3])
    x_2, y_2, z_2 = (
        heat_points[:, 0][:-1],
        heat_points[:, 1][:-1],
        heat_points[:, 3][:-1].reshape(32, 7).T[::-1, :],
    )
    # pos_2 = ax2.scatter(x_2, y_2, c=z_2, s=80, cmap="viridis_r")
    # fig.colorbar(pos_2, ax=ax2, label="different type of LoRA modules")
    color_list_2 = [
        (0.92941176, 0.49019608, 0.19215686),
        (0.58823529, 0.76470588, 0.49019608),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    color_list_2 = [
        (0.96862745, 0.76078431, 0.71764706),
        (0.8, 0.89411765, 0.74509804),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    cmap2 = colors.ListedColormap(color_list_2)
    im_2, cbar_2 = heatmap(
        z_1[4:, :],
        [
            "q_proj",
            "k_proj",
            "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax2,
        cmap=cmap1,
        cbarlabel="Rank Settting",
    )
    if cbar_2:
        cbar_2.set_ticks(ticks=[3, 5, 7], labels=["2", "4", "8"])
    # plt.suptitle("xxx")
    # ax3.set_title("3.0 Visualization of choices about the LoRA-like modules")
    # ax2.set_xlabel("transfomer block index")
    # ax2.set_ylabel("different linear module")
    ax3.set_xticks(range(0, heat_points[-1][0] + 1))
    ax3.set_yticks(range(1, len(ffn_type_dict)))
    # print(heat_points[:, 3])
    x_3, y_3, z_3 = (
        heat_points[:, 0][:-1],
        heat_points[:, 1][:-1],
        heat_points[:, 3][:-1].reshape(32, 7).T[::-1, :],
    )
    # pos_2 = ax2.scatter(x_2, y_2, c=z_2, s=80, cmap="viridis_r")
    # fig.colorbar(pos_2, ax=ax2, label="different type of LoRA modules")
    color_list_3 = [
        (0.92941176, 0.49019608, 0.19215686),
        (0.58823529, 0.76470588, 0.49019608),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    color_list_3 = [
        (0.96862745, 0.76078431, 0.71764706),
        (0.8, 0.89411765, 0.74509804),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    cmap3 = colors.ListedColormap(color_list_3)
    im_3, cbar_3 = heatmap(
        z_3[:4, :],
        [
            # "q_proj",
            # "k_proj",
            # "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax3,
        cmap=cmap3,
        cbarlabel="Type Settting",
    )
    if cbar_3:
        cbar_3.set_ticks(ticks=[1.3333, 2, 2.6667], labels=["L", "A", "D"])

    # ax4.set_title("Visualization of choices about the LoRA-like modules ")
    # ax2.set_xlabel("transfomer block index")
    # ax2.set_ylabel("different linear module")
    ax4.set_xticks(range(0, heat_points[-1][0] + 1))
    ax4.set_yticks(range(1, len(ffn_type_dict)))
    # print(heat_points[:, 3])
    x_4, y_4, z_4 = (
        heat_points[:, 0][:-1],
        heat_points[:, 1][:-1],
        heat_points[:, 3][:-1].reshape(32, 7).T[::-1, :],
    )
    # pos_2 = ax2.scatter(x_2, y_2, c=z_2, s=80, cmap="viridis_r")
    # fig.colorbar(pos_2, ax=ax2, label="different type of LoRA modules")
    color_list_4 = [
        (0.92941176, 0.49019608, 0.19215686),
        (0.58823529, 0.76470588, 0.49019608),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    color_list_4 = [
        (0.96862745, 0.76078431, 0.71764706),
        (0.8, 0.89411765, 0.74509804),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    cmap4 = colors.ListedColormap(color_list_4)
    im_4, cbar_4 = heatmap(
        z_3[4:, :],
        [
            "q_proj",
            "k_proj",
            "v_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax4,
        cmap=cmap4,
        cbarlabel="Type Settting",
    )
    if cbar_4:
        cbar_4.set_ticks(ticks=[1.3333, 2, 2.6667], labels=["L", "A", "D"])
    plt.subplots_adjust(wspace=0, hspace=0.4)
    # plt.grid()

    plt.savefig("logs/selection.pdf", format="pdf", bbox_inches="tight")
    # plt.tight_layout()
    # plt.savefig("logs/test.png")


def plot_heat_v2(heat_points):
    # plt.rcParams["font.family"] = "Helvetica"
    color_list = [(0.282, 0.792, 0.894), (0, 0.588, 0.78), (0, 0.243, 0.541)]
    cmap1 = colors.ListedColormap(color_list)
    # fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(15, 6))
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(20, 4))
    # ax1.set_title("Visualization of the rank choices about LoRA-like modules")
    # ax1.set_xlabel("transfomer block index")
    # ax1.set_ylabel("different rank of LoRA module")
    ax1.set_xticks(range(0, heat_points[-1][0] + 1))
    ax1.set_yticks(range(1, len(ffn_type_dict)))
    # pos_1 = ax1.scatter(heat_points[:, 0], heat_points[:, 1], c=heat_points[:, 2], s=80, cmap=cmap2)
    x_1, y_1, z_1 = (
        heat_points[:, 0][:-1].reshape(32, 7)[:, [0, 1, 2, 3, 4, 6]],
        heat_points[:, 1][:-1].reshape(32, 7)[:, [0, 1, 2, 3, 4, 6]],
        heat_points[:, 2][:-1].reshape(32, 7).T[::-1, :],
    )

    im_1, cbar_1 = heatmap(
        z_1[:, :],
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax1,
        cmap=cmap1,
        # cbar_kw={"ticks": [3, 5, 7], "labels": ["2", "4", "8"]},
        cbarlabel="Rank Setting",
    )
    if cbar_1:
        cbar_1.set_ticks(ticks=[3, 5, 7], labels=["2", "4", "8"])

    # plt.suptitle("xxx")
    # ax3.set_title("3.0 Visualization of choices about the LoRA-like modules")
    # ax2.set_xlabel("transfomer block index")
    # ax2.set_ylabel("different linear module")
    ax2.set_xticks(range(0, heat_points[-1][0] + 1))
    ax2.set_yticks(range(1, len(ffn_type_dict)))
    # print(heat_points[:, 3])
    x_3, y_3, z_3 = (
        heat_points[:, 0][:-1],
        heat_points[:, 1][:-1],
        heat_points[:, 3][:-1].reshape(32, 7).T[::-1, :],
    )
    # pos_2 = ax2.scatter(x_2, y_2, c=z_2, s=80, cmap="viridis_r")
    # fig.colorbar(pos_2, ax=ax2, label="different type of LoRA modules")
    color_list_3 = [
        (0.92941176, 0.49019608, 0.19215686),
        (0.58823529, 0.76470588, 0.49019608),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    color_list_3 = [
        (0.96862745, 0.76078431, 0.71764706),
        (0.8, 0.89411765, 0.74509804),
        (0.68627451, 0.86666667, 0.92941176),
    ]
    # color_list_3 = [
    #     (0.95294118, 0.96078431, 0.76078431),
    #     (0.8627451, 0.98039216, 0.87843137),
    #     (0.76862745, 0.92156863, 0.98039216),
    # ]
    cmap3 = colors.ListedColormap(color_list_3)
    im_3, cbar_3 = heatmap(
        z_3[:, :],
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ][::-1],
        [f"L{_}" for _ in range(1, 33)],
        ax=ax2,
        cmap=cmap3,
        cbarlabel="Type Settting",
    )
    if cbar_3:
        cbar_3.set_ticks(ticks=[1.3333, 2, 2.6667], labels=["L", "A", "D"])

    plt.subplots_adjust(wspace=0, hspace=0.4)
    # plt.grid()

    plt.savefig("logs/selection.svg", format="svg", bbox_inches="tight")
    # plt.tight_layout()
    # plt.savefig("logs/test.png")


# path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/naslora_v4/llama2/social_i_qa/independent/peft_naslora_lr1e-3_social_i_qa_independent_no_mlc_arch_lr005_60_20_22/adapter_masks.json"
# path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/naslora_v4/llama2/boolq/independent/peft_naslora_lr1e-3_boolq_independent_no_mlc_arch_lr002_pre_combibe_70_10/adapter_masks.json"
# # path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/naslora_v4/llama2/samsum/independent/peft_naslora_lr1e-3_independent_no_mlc_arch_lr001_60_20_22/adapter_masks.json"
search_space_path = (
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/config/search_space/common.yml"
)
# search_space_path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/config/search_space/adalora.yml"
# heat_points = get_points(path, search_space_path)
# # print(heat_points)
# plot_heat(heat_points)
path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/naslora_v4/llama2/boolq/independent/peft_naslora_lr1e-3_boolq_independent_no_mlc_arch_lr01_pre_combibe_70_10_steplr_10/adapter_masks.json"
# path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/naslora_v4/llama2/social_i_qa/independent/peft_naslora_lr1e-3_social_i_qa_independent_no_mlc_arch_lr01_pre_combibe_70_10_steplr_10/adapter_masks.json"
# heat_points = get_points(path, search_space_path)
heat_points = get_points(path, search_space_path)
# plot_heat(heat_points)
plot_heat_v2(heat_points)

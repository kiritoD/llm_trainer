import torch
import matplotlib.pyplot as plt
import numpy as np
#  for roberta
def hist(data, model_name, dt, alpha=0.8):
    keys = list(data.keys())
    title = f"Weight Analysis of {keys[0].split('_')[0]}"
    data = data.view(-1).numpy()
    if isinstance(data, dict):
        for key, value in data.items():
            plt.hist(value, bins=30, alpha=alpha, label=key)
    else:
        # 绘制直方图
        plt.hist(data, bins=30, color='skyblue', alpha=alpha)

    # 设置图表属性
    plt.title('weights')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    # 显示图表
    plt.savefig(f"/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/ditributions/hist_{model_name}_{dt}_{title}.jpg")
    plt.clf()

def heatmap(data, model_name, dt):
    keys = list(data.keys())
    title = f"Weight Analysis of {keys[0].split('_')[0]}"
    shape = list(list(data.values())[0].shape)
    length = len(data)
    x = int(np.sqrt(length))
    y = int(np.ceil(length / x))
    fig, axes = plt.subplots(x, y, sharex=True, sharey=True, figsize=(16, 16))
    fig.suptitle(title)
    
    vmin = 0
    vmid = 0
    for value in list(data.values()):
        values_ = torch.sort(value.view(-1))
        index_ = int(values_[0].shape[0] * 0.7)
        vmid_  = values_[0][index_]
        if vmid_ > vmid:
            vmid = vmid_
    images = []
    index = 0
    for key, value in data.items():
        
        if x == 1:
            im = axes[index].imshow(value, vmin=vmin, vmax=vmid*2, aspect='auto')
            images.append(im)
            axes[index].set_title(f"{key.split('_')[2]}")
        else:
            x_ = index // y
            y_ = index % y
            im = axes[x_][y_].imshow(value, vmin=vmin, vmax=vmid*2, aspect='auto')
            images.append(im)
            axes[x_][y_].set_title(f"{key.split('_')[2]}")
            
        
        index += 1
    fig.colorbar(images[0], ax=axes, orientation='horizontal', fraction=.1)
    plt.savefig(f"/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/ditributions/heatmap_{model_name}_{dt}_{title}.jpg")
    plt.clf()
    
def weights_plot(data, model_name, dt, alpha=0.2):
    heatmap(data, model_name, dt)
    
def weight_key_mapping(weight_data, model_name):
    key_mapping = {
        "roberta-base": "layer",
        "llama2": "layers"
    }
    module_index_mapping = {
        "roberta-base": 3,
        "llama2": 2
    }
    keys = list(weight_data.keys())

    layer_mapping = {}
    for item in keys:
        # print(item)
        if key_mapping[model_name] in item:
            key_suffixs = item.split(f"{key_mapping[model_name]}.")[-1].split(".")
            index = int(key_suffixs[0])
            module_name = key_suffixs[module_index_mapping[model_name]]
            # print(item)
            key = "_".join(["layer", str(index), module_name])
            lora_key = "a" if "lora_A" in item else "b"
            if key not in layer_mapping:
                layer_mapping[key] = {
                    lora_key: item
                }
            else:
                layer_mapping[key][lora_key] = item
    return layer_mapping

def weight_anaylsis(weight, dt="", model_name="roberta-base", interval=5):
    key_mapping = {
        "roberta-base": {
            "q": "query",
            "k": "key",
            'v': "value"
        },
        "llama2": {
            "q": "q_proj",
            "k": "k_proj",
            'v': "v_proj"
        }
    }
    layer_mapping = weight_key_mapping(weight, model_name)
    # for key, value in layer_mapping.items():
    #         # print(layer_mapping[key])
    #         weight_A = weight[value['a']]
    #         weight_B = weight[value['b']]
    #         m = torch.abs(weight_B @ weight_A)
    #         print(key, torch.mean(m), torch.max(m), torch.min(m), torch.std(m), torch.median(m))
    weights_query_dict = {}
    weights_value_dict = {}
    for key, value in layer_mapping.items():
        # print(layer_mapping[key])
        
        # print(key, torch.mean(m), torch.max(m), torch.min(m), torch.std(m))
        # print(key)
        index = int(key.split("_")[1])
        if index % interval == 0:
            if key_mapping[model_name]["q"] in key:
                # print(key_mapping[model_name]["q"], key, value['a'])
                weight_A = weight[value['a']].float()
                weight_B = weight[value['b']].float()
                m = torch.abs(weight_B @ weight_A)
                # print(value['a'], m.numel())
                weights_query_dict[f"query_{key}"] = m
            if key_mapping[model_name]["v"] in key:
                # print(key)
                weight_A = weight[value['a']].float()
                weight_B = weight[value['b']].float()
                m = torch.abs(weight_B @ weight_A)
                # print(value['a'], m.numel())
                weights_value_dict[f"value_{key}"] = m

    weights_plot(weights_query_dict, model_name=model_name, dt=dt, alpha=0.2)
    weights_plot(weights_value_dict, model_name=model_name, dt=dt, alpha=0.2)

# roberta
# roberta_weight_lora_path_mapping = {
#     "sst2": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/sst2/LORA/step_1320/adapter_model.bin",
#     "mnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/mnli/LORA/step_3068/adapter_model.bin",
#     "cola": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/cola/LORA/step_680/adapter_model.bin",
#     "mrpc": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/mrpc/LORA/step_300/adapter_model.bin",
#     "qnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/qnli/LORA/step_8200/adapter_model.bin",
#     "qqp": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/qqp/LORA/step_28440/adapter_model.bin",
#     "rte": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/rte/LORA/step_200/adapter_model.bin",
#     "stsb": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/stsb/LORA/step_460/adapter_model.bin",
#     "wnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/roberta/wnli/LORA/step_60/adapter_model.bin",
# }

# dt_name = ["sst2", "mnli", "cola", "mrpc", "qnli", "qqp", "rte", "stsb", "wnli"]
# # dt_name = ["sst2"]
# for dt in dt_name:
#     print(f"{dt}")
#     weight = torch.load(roberta_weight_lora_path_mapping[dt], map_location='cpu')
#     weight_anaylsis(weight, dt=dt, model_name="roberta-base", interval=1)

# llama2
llama2_weight_lora_path_mapping = {
    "sst2": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/sst2/LORA/step_780/adapter_model.bin",
    "mnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/mnli/LORA/step_780/adapter_model.bin",
    "cola": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/cola/LORA/step_670/adapter_model.bin",
    "mrpc": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/mrpc/LORA/step_285/adapter_model.bin",
    "qnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/qnli/LORA/step_780/adapter_model.bin",
    "qqp": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/qqp/LORA/step_780/adapter_model.bin",
    "rte": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/rte/LORA/step_195/adapter_model.bin",
    "stsb": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/stsb/LORA/step_450/adapter_model.bin",
    "wnli": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/wnli/LORA/step_50/adapter_model.bin",
    "all": "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/llama/all/LORA/step_4775/adapter_model.bin"
}

# dt_name = ["sst2", "mnli", "cola", "mrpc", "qnli", "qqp", "rte", "stsb", "wnli", "all"]
dt_name = ["all"]
for dt in dt_name:
    print(f"{dt}")
    weight = torch.load(llama2_weight_lora_path_mapping[dt], map_location='cpu')
    weight_anaylsis(weight, model_name="llama2", dt=dt,interval=2)

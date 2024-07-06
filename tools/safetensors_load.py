import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

model_path = '/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/test/mrpc_ep20_r1_relu/LORA_A/step_115/adapter_model.safetensors'

tensors = {}
with safe_open(model_path, framework="pt", device='cpu') as f:
    for k in f.keys():
        print(k)
        tensors[k] = f.get_tensor(k)

# print(tensors)
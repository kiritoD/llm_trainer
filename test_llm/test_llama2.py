# Load model directly
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def content_pre(content):
    for key in content:
        content[key] = torch.tensor([content[key]])


# tokenizer = AutoTokenizer.from_pretrained(
#     "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/hf_models/Llama-2-7b-hf"
# )
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/config/train/1b1gpt/hf_config"
)


def model_linear_postfix(model):
    name_arr = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name_arr.append(name)
    name_arr_new = list(set(map(lambda x: x.split(".")[-1], name_arr)))
    return name_arr_new


print(model_linear_postfix(model))

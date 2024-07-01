import os

import numpy as np
import torch

# from transformers import AutoModelForCausalLM

parameters_llama2_7B = 6857783827


def main(path):
    adapter_model = torch.load(path, map_location="cuda")
    adaper_model_parameters = sum([np.array(_.size()).cumprod()[-1] for _ in adapter_model.values()])
    print(f"{path}: {adaper_model_parameters:,}, percent:{adaper_model_parameters / parameters_llama2_7B:.3%}")


if __name__ == "__main__":
    index = 331
    adapter_path = f"/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/super_glue_all/adalora/adalora_super_glue_all_cosine_original_r8_4_42_1127/ADALORA/step_{index}/adapter_model.bin"
    main(adapter_path)

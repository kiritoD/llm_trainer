# A-LoRA

This is an implementation for A-LoRA. You can train an LLM with the following scripts. 

Attention: 

+ An environment with GPUs is necessary.
+ Bf16 for A100 and fp16 for V100
+ Deepspeed zero1-3 are all available.
+ Gradient checkpointing and accumulate strategies could be used if you don't have enough GPU memory.
+ If you have >1 GPUs, the shell will automatically use all the GPUs for training and inference.
+ If you have wandb, you can see all the training information in the wandb panel.

```shell
pip install -r requirements.txt
```

## Train

```shell
bash scripts/train.sh ./config/train/superglue/a-lora42.yml
```

## Inference

```shell
bash scripts/inference.py ./config/inference/llama/super_glue/sg_v1.yml
```

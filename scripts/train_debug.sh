#!/bin/bash
WORKER_GPU=8
# current dir peft_llm

torchrun --nproc_per_node $WORKER_GPU --nnodes 1 -m debugpy --connect 10.120.18.243:6036 train.py --config_path=$1
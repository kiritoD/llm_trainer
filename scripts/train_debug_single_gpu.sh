#!/bin/bash
# current dir peft_llm
export CUDA_VISIBLE_DEVICES=0
DEFAULT_PORT=29500
PORT=${2:-$DEFAULT_PORT}
torchrun --nproc_per_node 1 --nnodes 1 --master_port $PORT -m debugpy --connect 10.120.18.243:6036 train.py --config_path=$1
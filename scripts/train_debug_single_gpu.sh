#!/bin/bash
# current dir peft_llm

torchrun --nproc_per_node 1 --nnodes 1 -m debugpy --connect 10.120.18.243:6032 train.py --config_path=$1
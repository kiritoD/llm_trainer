#!/bin/bash
WORKER_GPU=$1
# current dir peft_llm
# export CUDA_VISIBLE_DEVICES=2,3,6,7

DEFAULT_PORT=29500
PORT=${3:-$DEFAULT_PORT}
# torchrun --nproc_per_node $WORKER_GPU --nnodes $WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $WORKER_0_HOST --master_port $PORT train.py --config_path=$2
torchrun --nproc_per_node $WORKER_GPU --master_port $PORT --nnodes 1 train.py --config_path=$2
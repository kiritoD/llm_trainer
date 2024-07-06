#!/bin/bash
set -e
# bash scripts/train.sh 
# path=/mnt/bn/target_dir/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/config/train/lora/r2

function start_train() {
    echo "start to train, the command script: bash scripts/train.sh $1 $2"
    bash scripts/train.sh $1 $2
    sleep 2
}

function readDir() {
  local dir=$2
  local files
  files=$(ls "$dir")
  for file in $files; do
    local path="$dir/$file"
    if [ -d "$path" ]; then
      readDir "$path"
    else
      start_train $1 $path && echo 'end'
    fi
  done
}
dir=$2
worker_gpu=$1
readDir "$worker_gpu" "$dir"
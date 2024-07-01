#!/bin/bash
set -e
# bash scripts/train.sh 
# path=/mnt/bn/target_dir/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/config/train/lora/r2

start_train() {
    echo "start to train, the command script: bash scripts/train.sh $1"
    bash scripts/train.sh $1
    sleep 2
}

readDir() {
  local dir=$1
  local files
  files=$(ls "$dir")
  for file in $files; do
    local path="$dir/$file"
    if [ -d "$path" ]; then
      readDir "$path"
    else
      start_train $path && echo 'end'
    fi
  done
}
readDir $1
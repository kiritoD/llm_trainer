#!/bin/bash
PORTS=$WORKER_0_PORT 
PORT=(${PORTS//,/ })
# current dir peft_llm
remote_path=target_dir_remote
local_path=target_dir

mode=$1

if [ ${mode} != "local" ]
then
    content=$(cat $2 | grep $remote_path)
    if [ "${content}" == "" ]
    then
    sh scripts/replace_for_yaml.sh $2 ${local_path} ${remote_path}
    fi
else
    sh scripts/replace_for_yaml.sh $2 ${remote_path} ${local_path}
fi

torchrun --nproc_per_node $WORKER_GPU --nnodes $WORKER_NUM --node_rank=$ARNOLD_ID --master_addr $WORKER_0_HOST --master_port $PORT inference.py --config_path=$2

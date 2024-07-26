import subprocess
import sys
from src.utils.mp_tool import MP_Tool
import os
import time
import multiprocessing as mp
import json
import glob

FILE_LOCK = mp.Lock()
log_dir = "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/model_output/logs/"
has_run = [
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dtcola_ep0080_lh001_lr0002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dtmrpc_ep0030_lh001_lr0002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dtstsb_ep0080_lh001_lr0002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dtqnli_ep0025_lh001_lr0002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dtsst2_ep0060_lh0004_lr0002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr64r16/dt0rte_ep00160_lh0004_lr002_hr0064_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr8r8/dtsst2_ep0060_lh0004_lr0002_hr0008_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr8r8/dt0rte_ep00160_lh0004_lr002_hr0008_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr8r8/dtqnli_ep0025_lh001_lr0002_hr0008_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/011_for_glue/hr8r8/dtcola_ep0080_lh001_lr0002_hr0008_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dt0rte_ep00160_lh0004_lr002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dtmrpc_ep0030_lh001_lr0002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dtcola_ep0080_lh001_lr0002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dtsst2_ep0060_lh0004_lr0002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dtqnli_ep0025_lh001_lr0002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r8/dtstsb_ep0080_lh001_lr0002_hr0128_r0008.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r16/dtstsb_ep0080_lh001_lr0002_hr0128_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r16/dt0rte_ep00160_lh0004_lr002_hr0128_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r16/dtmrpc_ep0030_lh001_lr0002_hr0128_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr128r16/dtqnli_ep0025_lh001_lr0002_hr0128_r0016.yml",
    # "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/config/hssa/train/distribution_analysis/roberta_base/explore/010_for_glue/hr64r16/dt0rte_ep00160_lh0004_lr002_hr0064_r0016.yml",
]


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


output = dict(start_time=get_local_time())


def json_write(file_path, content: dict):
    with open(file_path, "w") as json_file:
        json.dump(content, json_file)


def json_read(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def output_file(file_name, content):
    FILE_LOCK.acquire()
    output_dict = json_read(file_name)
    output_dict.update(content)
    json_write(file_name, output_dict)
    FILE_LOCK.release()


def train_cmd(yaml_file, gpu_number, port_id, log_file_name, number):
    global log_dir
    info_dict = {yaml_file: dict(start_time=get_local_time())}
    output_file(log_file_name, info_dict)
    start_time = time.time()
    cmd = f"bash scripts/train.sh {gpu_number} {yaml_file} 28{port_id:0>3}"
    print(f"[{number}]: start training...\n (cmd: {cmd})")
    yaml_file_paths = yaml_file.split("/")
    yaml_file_paths[-1] = yaml_file_paths[-1].replace(".yml", ".log")
    std_file = os.path.join(
        log_dir, yaml_file_paths[-3], yaml_file_paths[-2], yaml_file_paths[-1]
    )
    os.makedirs(os.path.dirname(std_file), exist_ok=True)
    std_ = open(std_file, "w")
    subprocess.run(cmd.split(), stdout=std_, stderr=std_)
    # subprocess.run(["sleep", f"{port_id}"])
    # subprocess.run(["ls"], stdout=std_, stderr=std_)
    end_time = time.time()
    info_dict[yaml_file].update(dict(end_time=get_local_time()))
    info_dict[yaml_file].update(dict(train_time=f"{end_time - start_time:.2f}"))
    output_file(log_file_name, info_dict)
    print(f"[{number}]: end training~")


def get_all_yaml_files(config_dirs):
    config_files = []
    if isinstance(config_dirs, str):
        config_files = glob.glob(os.path.join(config_dirs, "**/*.yml"), recursive=True)
    else:
        for config_dir in config_dirs:
            config_files.extend(
                glob.glob(os.path.join(config_dir, "**/*.yml"), recursive=True)
            )

    return config_files


def main(log_file_name, config_dirs, processing_number, gpu_number=1):
    mp_tool = MP_Tool(processing_number)
    print(config_dirs)
    config_files = get_all_yaml_files(config_dirs)
    results = []
    port_ids = list(range(processing_number))
    print(len(config_files), port_ids)
    for i, config_file in enumerate(config_files):
        if config_file in has_run:
            continue
        result = mp_tool.create(
            train_cmd,
            True,
            config_file,
            gpu_number,
            i,
            log_file_name,
            i,
        )
        results.append(result)

    for _ in results:
        _.wait()
    mp_tool.close()
    subprocess.run(["python", "scripts/gpu_occ.py"])


if __name__ == "__main__":
    assert (
        len(sys.argv) >= 3
    ), f"plese input the name of output log file, the `cmd` can be `python xx.py log_file_name processing_number config_dir1 config_dir2 config_dir3` and the output dir in `{log_dir}`"
    print(sys.argv)
    log_file_name = sys.argv[1]
    log_file_path = os.path.join(log_dir, log_file_name)
    processing_number = int(sys.argv[2])
    config_dirs = sys.argv[3:]
    json_write(log_file_path, output)
    main(log_file_path, config_dirs, processing_number)

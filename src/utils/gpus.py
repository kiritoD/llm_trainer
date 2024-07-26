import shlex
import subprocess


def get_memory_free() -> bool:
    output = subprocess.check_output(
        shlex.split("nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader")
    )

    memory_usage = output.decode().split("\n")
    memory_usage = [int(m) for m in memory_usage if m != ""]
    return memory_usage[0]

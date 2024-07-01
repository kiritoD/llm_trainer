import logging
import sys

import torch.distributed as dist


def rank_zero_info(content: str, logger, print_type: str = "info"):
    output_method = getattr(logger, print_type)
    try:
        if dist.get_rank() == 0:
            output_method(content)
    except:
        output_method(content)


def get_logger(name: str):
    # logger initialize
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    # add handler
    logger.addHandler(handler)

    return logger

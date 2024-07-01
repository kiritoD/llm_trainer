import glob
import sys
import os
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(dir_path: str, target_file: str):
    """clean `target_file` recursive in a directory

    Parameters
    ----------
    dir_path : str
        the path of directory
    target_file : str
        the target file name

    examples:
    >>> python pytorch_clean.py outputs pytorch_model.bin
    Raises
    ------
    OSError
        _description_
    """
    if not os.path.isdir(dir_path):
        raise OSError(f"{dir_path} is not a directory")
    target_files = glob.glob(f"{dir_path}/**/{target_file}", recursive=True)
    for file_path in target_files:
        logger.info(f"start to remove the {file_path}~")
        os.remove(file_path)
    logger.info(f"total files: {len(target_files)}")


if __name__ == "__main__":
    dir_path = sys.argv[1]
    taget_file = sys.argv[2]
    main(dir_path, taget_file)

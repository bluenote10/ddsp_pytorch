import os
from pathlib import Path


def get_train_dir() -> Path:
    return Path(os.environ["TRAIN_DIR"])


def get_data_dir() -> Path:
    return Path(os.environ["DATA_DIR"])

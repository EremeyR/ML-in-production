import json
from utils import CONFIG_PATH
import pandas as pd
import numpy as np


class Args:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as fh:
        config = json.load(fh)

    model_type = config["model_type"]
    n_estimators = config["n_estimators"]
    n_jobs = config["n_jobs"]
    train_size = config["train_size"]
    model_path = config["model_path"]
    model_name = config["model_name"]
    categorical_cols = config["categorical_cols"]
    solution_path = config["solution_path"]
    solution_name = config["solution_name"]
    data_path = config["data_path"]
    data_name = config["data_name"]


def create_labels(size):
    return np.random.randint(0, 2, size)

import json
from utils import CONFIG_PATH
import pandas as pd
import numpy as np


class Args:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as fh:
        config = json.load(fh)

    model_type = config["model_type"]
    n_estimators = config["n_estimators"]
    random_state = config["random_state"]
    n_jobs = config["n_jobs"]
    train_size = config["train_size"]
    model_path = config["model_path"]
    model_name = config["model_name"]
    solution_path = config["solution_path"]
    solution_name = config["solution_name"]
    data_path = config["data_path"]
    data_name = config["data_name"]


def create_features(size):
    df = pd.DataFrame()
    df["age"] = np.random.randint(30, 90, size)
    df["sex"] = np.random.randint(0, 2, size)
    df["cp"] = np.random.randint(0, 4, size)
    df["trestbps"] = np.random.randint(90, 200, size)
    df["chol"] = np.random.randint(170, 200, size)
    df["fbs"] = np.random.randint(0, 2, size)
    df["restecg"] = np.random.randint(0, 3, size)
    df["thalach"] = np.random.randint(90, 200, size)
    df["exang"] = np.random.randint(0, 2, size)
    df["oldpeak"] = np.random.uniform(0, 5, size)
    df["slope"] = np.random.randint(0, 2, size)
    df["ca"] = np.random.randint(0, 4, size)
    df["thal"] = np.random.randint(0, 3, size)
    return df


def create_labels(size):
    return np.random.randint(0, 2, size)

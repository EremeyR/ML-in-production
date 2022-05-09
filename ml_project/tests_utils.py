import utils

import pandas as pd
import numpy as np


class Args:
    model_type = utils.model_type
    n_estimators = utils.n_estimators
    random_state = utils.random_state
    n_jobs = utils.n_jobs
    train_size = utils.train_size
    model_path = utils.model_path
    model_name = utils.model_name
    solution_path = utils.solution_path
    solution_name = utils.solution_name
    data_path = utils.data_path
    data_name = utils.data_name


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

from datetime import timedelta
import os
import pathlib

import pandas as pd
import numpy as np


from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


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
    df = pd.DataFrame()
    df["condition"] = np.random.randint(0, 2, size)
    return df


# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(21),
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def save_to_dir(path: str, name: str, data: pd.DataFrame) -> None:
    if os.path.exists(f"{path}/{name}.csv"):
        old_data = pd.read_csv(f"{path}/{name}.csv")
        data = pd.concat([old_data, data], ignore_index=True)
    data.to_csv(f"{path}/{name}.csv", index_label=False, index=False)


def generator(
    year: str,
    month: str,
    day: str,
    output_dir: str):
    pathlib.Path(f"{output_dir}/raw/{year}_{month}_{day}").mkdir(parents=True,
                                                                 exist_ok=True)

    data_size = np.random.randint(10, 30)

    new_data = create_features(data_size)
    new_target = create_labels(data_size)

    save_to_dir(f"{output_dir}/raw/{year}_{month}_{day}", "data", new_data)
    save_to_dir(f"{output_dir}/raw/{year}_{month}_{day}", "target", new_target)


with DAG(
    dag_id="01_generate_data",
    default_args=default_args,
    schedule_interval=timedelta(hours=4),
) as dag:

    generator = PythonOperator(
        task_id="generator",
        python_callable=generator,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "output_dir": "/opt/airflow/data",
        }
    )

    generator

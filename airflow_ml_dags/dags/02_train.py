from datetime import timedelta
import pathlib
import os

import pickle

import pandas as pd

from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.sensors.python import PythonSensor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(21),
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

default_data_dict = {
    "year": "{{ execution_date.year }}",
    "month": "{{ execution_date.month }}",
    "day": "{{ execution_date.day }}",
    "output_dir": "/opt/airflow/data",
}


def _wait_for_data(year: str, month: str, day: str, output_dir: str):
    return os.path.exists(f"{output_dir}/raw/{year}_{month}_{day}/data.csv")


def preprocess(year: str, month: str, day: str, output_dir: str):
    data = pd.read_csv(f"{output_dir}/raw/{year}_{month}_{day}/data.csv")
    target = pd.read_csv(f"{output_dir}/raw/{year}_{month}_{day}/target.csv")

    pathlib.Path(f"{output_dir}/processed/{year}_{month}_{day}").mkdir(
        parents=True, exist_ok=True)
    data.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/data.csv",
                index_label=False, index=False)
    target.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/target.csv",
                  index_label=False, index=False)


def split(year: str, month: str, day: str, output_dir: str):
    data = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/data.csv")
    target = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/target.csv")

    data_train, data_test, target_train, target_test = \
        train_test_split(data, target, random_state=42)

    data_train.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/data_train.csv",
                      index_label=False, index=False)
    data_test.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/data_test.csv",
                     index_label=False, index=False)
    target_train.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/target_train.csv",
                        index_label=False, index=False)
    target_test.to_csv(f"{output_dir}/processed/{year}_{month}_{day}/target_test.csv",
                       index_label=False, index=False)


def train(year: str, month: str, day: str, output_dir: str):
    model = RandomForestRegressor(random_state=42)

    features = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/data_train.csv")
    target = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/target_train.csv")

    model.fit(features, target)

    pathlib.Path(f"{output_dir}/models/{year}_{month}_{day}").mkdir(parents=True,
                                                                       exist_ok=True)

    with open(f"{output_dir}/models/{year}_{month}_{day}/model.pickle", "wb") as f:
        pickle.dump({"model": model, "f1_score": -1.}, f)


def validate(year: str, month: str, day: str, output_dir: str):
    with open(f"{output_dir}/models/{year}_{month}_{day}/model.pickle", "rb") as f:
        model = pickle.load(f)["model"]

    features = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/data_test.csv")
    target = pd.read_csv(f"{output_dir}/processed/{year}_{month}_{day}/target_test.csv")

    metric = f1_score(target, model.predict(features).round())

    with open(f"{output_dir}/models/{year}_{month}_{day}/model.pickle", "wb") as f:
        pickle.dump({"model": model, "f1_score": metric}, f)


with DAG(
    dag_id="02_train",
    default_args=default_args,
    schedule_interval=timedelta(days=7),
) as dag:
    wait = PythonSensor(
        task_id="wait_for_file",
        python_callable=_wait_for_data,
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
        op_kwargs=default_data_dict
    )

    processing = PythonOperator(
        task_id="processing",
        python_callable=preprocess,
        op_kwargs=default_data_dict
    )

    splitting = PythonOperator(
        task_id="splitting",
        python_callable=split,
        op_kwargs=default_data_dict
    )

    training = PythonOperator(
        task_id="training",
        python_callable=train,
        op_kwargs=default_data_dict
    )

    validating = PythonOperator(
        task_id="validating",
        python_callable=validate,
        op_kwargs=default_data_dict
    )

    wait >> processing >> splitting >> training >> validating

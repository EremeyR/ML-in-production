from datetime import timedelta
import pathlib

import pickle

import pandas as pd

from airflow import DAG

from airflow.models import Variable

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(21),
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def use_model(year: str, month: str, day: str, output_dir: str):
    pathlib.Path(f"{output_dir}/predictions/{year}_{month}_{day}").mkdir(
        parents=True, exist_ok=True)

    model_path = Variable.get("current_model_path")
    with open(f"{output_dir}/{model_path}", "rb") as f:
        model = pickle.load(f)["model"]

    data = pd.read_csv(f"{output_dir}/raw/{year}_{month}_{day}/data.csv")

    pd.DataFrame(model.predict(data).round()).to_csv(
        f"{output_dir}/predictions/{year}_{month}_{day}/predictions.csv")


with DAG(
    dag_id="03_model_using",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
) as dag:
    using = PythonOperator(
        task_id="using",
        python_callable=use_model,
        op_kwargs={
            "year": "{{ execution_date.year }}",
            "month": "{{ execution_date.month }}",
            "day": "{{ execution_date.day }}",
            "output_dir": "/opt/airflow/data",
        }
    )

    using

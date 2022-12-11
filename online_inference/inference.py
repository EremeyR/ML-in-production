import pickle

from typing import List

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import pandas as pd

import os
import sys
sys.path.append('..')

app = FastAPI()

model = None


class HeartFailurePredictionModel(BaseModel):
    data: List[List[float]]
    features: List[str]


class HeartFailurePrediction(BaseModel):
    label: float


@app.on_event("startup")
def load_model():

    # model_path = os.getenv("PATH_TO_MODEL")
    model_path = "./model.pickle"

    try:
        with open(model_path, 'rb') as f:
            global model
            model = pickle.load(f)
    except Exception as exception:
        logging.error(f"model loading error: {exception}")
        raise OSError(f"model loading error: {exception}")


@app.get("/predict", response_model=List[HeartFailurePrediction])
def predict(request: HeartFailurePredictionModel = None):
    request_df = pd.DataFrame(request.data, columns=request.features)
    return [HeartFailurePrediction(label=model.predict(request_df))]


@app.get("/health")
def health():
    if not model:
        raise HTTPException(status_code=400, detail="Model not found")


@app.put("/startup")
def health():
    """use for tests"""
    load_model()

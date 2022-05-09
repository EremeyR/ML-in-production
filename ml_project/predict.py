import logging
import pandas as pd

from utils import args_parser, load_model, safe_solution
from prepare import get_dataset


def predict_model(model, features: pd.DataFrame):
    return model.predict(features)


def predict(argues):
    model = load_model(argues)

    _, features, _, _ = get_dataset(argues)

    predicts = predict_model(model, features)

    safe_solution(predicts, argues)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    predict(args)
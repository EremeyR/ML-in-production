from utils import args_parser, save_model
from prepare import get_dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import logging


def train_model(features: pd.DataFrame, target: pd.DataFrame, argues):
    if argues.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=argues.n_estimators, random_state=argues.random_state,
            n_jobs=argues.n_jobs
        )
        logging.info("RandomForestRegressor was initialized")
    elif argues.model_type == "LinearRegression":
        model = LinearRegression()
        logging.info("RandomForestRegressor was initialized")
    else:
        raise NotImplementedError()
    logging.info("Fitting was started")

    model.fit(features, target)
    logging.info("Fitting was ended")
    return model


def train(argues):
    x_train, _, y_train, _ = get_dataset(argues)
    logging.info("Dataset were gotten")

    model = train_model(x_train, y_train, argues)
    logging.info("Model was obtained")

    save_model(model, argues)
    logging.info("Model was saved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    train(args)



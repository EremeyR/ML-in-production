from utils import args_parser, save_model, load_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd
import logging


def train_model(features, target, model_type, n_estimators, random_state):
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        logging.info("RandomForestRegressor was initialized")
    elif model_type == "LinearRegression":
        model = LinearRegression()
        logging.info("RandomForestRegressor was initialized")
    else:
        logging.error("Incorrect model type")
        raise NotImplementedError()
    logging.info("Fitting was started")

    model.fit(features, target)
    logging.info("Fitting was ended")
    return model


def train(argues):
    x_train = load_data("x_train", argues.data_path)
    y_train = load_data("y_train", argues.data_path)
    logging.info("Dataset were gotten")

    model = train_model(x_train, y_train, argues.model_type, argues.n_estimators, argues.random_state)
    logging.info("Model was obtained")

    save_model(model, argues.model_path, argues.model_name)
    logging.info("Model was saved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    train(args)



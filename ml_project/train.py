from utils import args_parser, save_model, load_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import logging

import sys
sys.path.append('..')

from common_artifacts.utils import OHETransformer


def train_model(features, target, model_type, n_estimators,
                categorical_cols: str):
    if model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42
        )
        logging.info("RandomForestRegressor was initialized")
    elif model_type == "LinearRegression":
        model = LinearRegression()
        logging.info("RandomForestRegressor was initialized")
    else:
        logging.error("Incorrect model type")
        raise NotImplementedError()
    logging.info("Fitting was started")

    if categorical_cols != "":
        model = Pipeline(steps=[
                       ('experimental_trans', OHETransformer(
                           categorical_cols.split())),
                       ('linear_model', LinearRegression())
        ])

    try:
        model.fit(features, target)
    except KeyError:
        logging.error("Incorrect column name")
        raise KeyError("Incorrect column name")

    logging.info("Fitting was ended")
    return model


def train(argues):
    x_train = load_data("x_train", argues.data_path)
    y_train = load_data("y_train", argues.data_path)
    logging.info("Dataset were gotten")

    model = train_model(x_train, y_train, argues.model_type,
                        argues.n_estimators, argues.categorical_cols)
    logging.info("Model was obtained")

    save_model(model, argues.model_path, argues.model_name)
    logging.info("Model was saved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    train(args)

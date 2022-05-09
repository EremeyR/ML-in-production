import logging
import pandas as pd

from utils import args_parser, load_model, save_solution, load_data


def predict_model(model, features: pd.DataFrame):
    return model.predict(features)


def predict(argues):
    model = load_model(argues)
    logging.info("Model was loaded")

    features = load_data("x_test", argues)
    logging.info("Features were loaded")

    predicts = predict_model(model, features)
    logging.info("Features were obtained")

    save_solution(predicts, argues)
    logging.info("Solution was saved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    predict(args)

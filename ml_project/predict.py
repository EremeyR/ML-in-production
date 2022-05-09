import logging
import pandas as pd

from utils import args_parser, load_model, save_solution, load_data


def predict_model(model, features: pd.DataFrame):
    return model.predict(features)


def predict(argues):
    model = load_model(argues.model_path, argues.model_name)
    logging.info("Model was loaded")

    features = load_data("x_test", argues.data_path)
    logging.info("Features were loaded")

    predicts = predict_model(model, features)
    logging.info("Features were obtained")

    save_solution(predicts, argues.solution_path, argues.solution_name)
    logging.info("Solution was saved")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    predict(args)

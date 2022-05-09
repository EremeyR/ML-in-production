from utils import args_parser
from prepare import get_dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas


def train_model(features: pandas.DataFrame, target: pandas.DataFrame, argues):
    if argues.model_type == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=argues.n_estimators, random_state=argues.random_state,
            n_jobs=argues.n_jobs
        )
    elif argues.model_type == "LinearRegression":
        model = LinearRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def train(argues):
    x_train, _, y_train, _ = get_dataset(argues)
    model = train_model(x_train, y_train, argues)


if __name__ == '__main__':
    args = args_parser()
    train(args)



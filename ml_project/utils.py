import argparse
import pickle
import numpy as np

from config import model_type, n_estimators, random_state, n_jobs, train_size,\
    model_path, model_name, data_path, data_name, solution_path, solution_name


def args_parser():
    parser = argparse.ArgumentParser(description='Listen Attend and Spell')
    # general
    parser.add_argument('--model-type', type=str, default=model_type, help='model type')
    parser.add_argument('--n-estimators', type=int, default=n_estimators, help='number of trees')
    parser.add_argument('--random-state', type=int, default=random_state, help=' sklearn random state')
    parser.add_argument('--n-jobs', type=int, default=n_jobs, help='n_jobs')
    parser.add_argument('--train-size', type=float, default=train_size, help='train size')
    parser.add_argument('--model-path', type=str, default=model_path, help='model path')
    parser.add_argument('--model-name', type=str, default=model_name, help='model name')
    parser.add_argument('--solution-path', type=str, default=solution_path, help='solution path')
    parser.add_argument('--solution-name', type=str, default=solution_name, help='solution name')
    parser.add_argument('--data-path', type=str, default=data_path, help='data path')
    parser.add_argument('--data-name', type=str, default=data_name, help='data name')
    args = parser.parse_args()
    return args


def save_model(model, argues):
    with open(f'{argues.model_path}/{argues.model_name}.pickle', 'wb') as f:
        pickle.dump(model, f)


def load_model(argues):
    with open(f'{argues.model_path}/{argues.model_name}.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(data_type: str, argues):
    if data_type not in ["x_train", "x_test", "y_train", "y_test"]:
        raise "Unknown dataset type : {}".format(data_type)

    with open(f'{argues.data_path}/prepared_data.pickle', 'rb') as f:
        data = pickle.load(f)
    return data[data_type]


def safe_solution(predicts: np.ndarray, argues) -> None:
    with open(f"{argues.solution_path}/{argues.solution_name}", 'w') as fout:
        print('Id', 'Prediction', sep=',', file=fout)
        for i, prediction in enumerate(predicts):
            print(i, int(prediction), sep=',', file=fout)


def save_prepared_data(x_train, x_test, y_train, y_test, argues):
    with open(f'{argues.data_path}/prepared_data.pickle', 'wb') as f:
        pickle.dump({"x_train": x_train, "x_test": x_test,
                     "y_train": y_train, "y_test": y_test}, f)

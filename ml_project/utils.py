import argparse
import pickle

from config import model_type, n_estimators, random_state, n_jobs, train_size,\
    model_path, model_name, data_path


def args_parser():
    parser = argparse.ArgumentParser(description='Listen Attend and Spell')
    # general
    parser.add_argument('--model_type', type=str, default=model_type, help='model type')
    parser.add_argument('--n_estimators', type=int, default=n_estimators, help='number of trees')
    parser.add_argument('--random_state', type=int, default=random_state, help=' sklearn random state')
    parser.add_argument('--n_jobs', type=int, default=n_jobs, help='n_jobs')
    parser.add_argument('--train_size', type=float, default=train_size, help='train size')
    parser.add_argument('--model_path', type=str, default=model_path, help='model path')
    parser.add_argument('--model_name', type=str, default=model_name, help='model name')
    parser.add_argument('--data_path', type=str, default=data_path, help='data path')
    args = parser.parse_args()
    return args


def save_model(model, argues):
    with open(f'{argues.data_path}/{argues.data_name}.pickle', 'wb') as f:
        pickle.dump(model, f)


def load_model(argues):
    with open(f'{argues.data_path}/{argues.data_name}.pickle', 'wb') as f:
        model = pickle.load(f)

    return model

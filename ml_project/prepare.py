import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import save_prepared_data, args_parser


def get_dataset(argues):
    df = pd.read_csv(f"{argues.data_path}/{argues.data_name}")

    y = df[df.columns[-1]]
    del df[df.columns[-1]]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=argues.train_size,
                                                        random_state=argues.random_state)
    return x_train, x_test, y_train, y_test


def prepare_data(argues):
    x_train, x_test, y_train, y_test = get_dataset(argues)
    logging.info(f"Features and labels were gotten from {argues.data_name}")
    save_prepared_data(x_train, x_test, y_train, y_test, argues)
    logging.info(f"Features and labels were saved to {argues.data_path}/prepared_data.pickle")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    prepare_data(args)

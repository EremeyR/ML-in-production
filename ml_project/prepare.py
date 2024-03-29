import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import save_prepared_data, args_parser


def load_dataset(path, name, trn_size):
    try:
        df = pd.read_csv(f"{path}/{name}")
    except Exception:
        logging.error("dataset loading error")
        raise OSError("dataset loading error")

    y = df[df.columns[-1]]
    del df[df.columns[-1]]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=trn_size,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def prepare_data(argues):
    x_train, x_test, y_train, y_test = load_dataset(argues.data_path,
                                                    argues.data_name,
                                                    argues.train_size)
    logging.info(f"Features and labels were gotten from {argues.data_name}")
    save_prepared_data(x_train, x_test, y_train, y_test, argues.data_path)
    logging.info(f"Features and labels were saved to {argues.data_path}"
                 f"/prepared_data.pickle")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = args_parser()
    prepare_data(args)

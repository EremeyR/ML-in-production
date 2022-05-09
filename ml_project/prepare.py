import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(args):
    df = pd.read_csv(args.data_path)

    y = df[df.columns[-1]]
    del df[df.columns[-1]]
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=args.train_size,
                                                        random_state=args.random_state)

    return x_train, x_test, y_train, y_test

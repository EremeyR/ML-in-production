import unittest
from tests_utils import Args, create_features, create_labels

import train as trn
import prepare as prp
import predict as prd
import utils as utl


class TestStringMethods(unittest.TestCase):
    def test_prepare(self):
        args = Args()
        prp.prepare_data(args)

    def test_train(self):
        args = Args()
        trn.train(args)

    def test_predict(self):
        args = Args()
        prp.prepare_data(args)
        prd.predict(args)

    def test_load_dataset(self):
        args = Args()
        prp.load_dataset(args.data_path, args.data_name, args.train_size)

    def test_train_model(self):
        args = Args()
        features = create_features(100)
        labels = create_labels(100)
        trn.train_model(features, labels, args.model_type, args.n_estimators)

    def test_train_model_with_short_data(self):
        args = Args()
        features = create_features(1)
        labels = create_labels(1)
        trn.train_model(features, labels, args.model_type, args.n_estimators)

    def test_predict_model(self):
        args = Args()
        features = create_features(100)
        model = utl.load_model(args.model_path, args.model_name)
        prd.predict_model(model, features)

    def test_predict_model_with_short_data(self):
        args = Args()
        features = create_features(1)
        model = utl.load_model(args.model_path, args.model_name)
        prd.predict_model(model, features)

    def test_prepare_train_predict_RandomForest(self):
        x_train, x_test, y_train, y_test = \
            prp.load_dataset("data", "heart_cleveland_upload.csv", 0.67)
        model = trn.train_model(x_train, y_train, "RandomForestRegressor",
                                100)
        predicts = prd.predict_model(model, x_test)
        utl.save_solution(predicts, ".", "RandomForest solution")

    def test_prepare_train_predict_LinearRegression(self):
        x_train, x_test, y_train, y_test = \
            prp.load_dataset("data", "heart_cleveland_upload.csv", 0.67)
        model = trn.train_model(x_train, y_train, "LinearRegression",
                                None)
        predicts = prd.predict_model(model, x_test)
        utl.save_solution(predicts, ".", "RandomForest solution")


if __name__ == '__main__':
    unittest.main()

import unittest
from tests_utils import Args, create_labels

import train as trn
import prepare as prp
import predict as prd
import utils as utl

import sys
sys.path.append('..')
from common_artifacts.utils import OHETransformer, create_features


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
        trn.train_model(features, labels, args.model_type, args.n_estimators,
                        "sex cp fbs restecg exang slope ca thal")

    def test_train_model_with_short_data(self):
        args = Args()
        features = create_features(1)
        labels = create_labels(1)
        trn.train_model(features, labels, args.model_type, args.n_estimators,
                        "sex cp fbs restecg exang slope ca thal")

    def test_predict_model(self):
        train_features = create_features(100)
        labels = create_labels(100)
        model = trn.train_model(train_features, labels, "LinearRegression",
                                0, "sex cp fbs restecg exang slope ca thal")
        test_features = create_features(50)
        prd.predict_model(model, test_features)

    def test_predict_model_without_transformer(self):
        train_features = create_features(100)
        labels = create_labels(100)
        model = trn.train_model(train_features, labels, "LinearRegression",
                                0, "")
        test_features = create_features(50)
        prd.predict_model(model, test_features)

    def test_predict_model_with_short_data(self):
        train_features = create_features(100)
        labels = create_labels(100)
        model = trn.train_model(train_features, labels, "LinearRegression",
                                0, "")
        test_features = create_features(1)
        prd.predict_model(model, test_features)

    def test_prepare_train_predict_RandomForest(self):
        x_train, x_test, y_train, y_test = \
            prp.load_dataset("data", "heart_cleveland_upload.csv", 0.67)
        model = trn.train_model(x_train, y_train, "RandomForestRegressor",
                                100, "sex cp fbs restecg exang slope ca thal")
        predicts = prd.predict_model(model, x_test)
        utl.save_solution(predicts, ".", "RandomForest solution")

    def test_prepare_train_predict_LinearRegression(self):
        x_train, x_test, y_train, y_test = \
            prp.load_dataset("data", "heart_cleveland_upload.csv", 0.67)
        model = trn.train_model(x_train, y_train, "LinearRegression",
                                0, "sex cp fbs restecg exang slope ca thal")
        predicts = prd.predict_model(model, x_test)
        utl.save_solution(predicts, ".", "RandomForest solution")

    def test_transformer_with_one_col(self):
        features = create_features(1000)
        transformer = OHETransformer(["sex"])
        transformed_features = transformer.fit_transform(features)

        self.assertEqual(features.shape[0], transformed_features.shape[0])
        self.assertEqual(features.shape[1], transformed_features.shape[1] - 1)

        transformer = OHETransformer(["cp"])
        transformed_features = transformer.fit_transform(features)

        self.assertEqual(features.shape[0], transformed_features.shape[0])
        self.assertEqual(features.shape[1], transformed_features.shape[1] - 3)

    def test_transformer_with_several_cols(self):
        features = create_features(100)
        transformer = OHETransformer(["sex", "cp"])
        transformed_features = transformer.fit_transform(features)

        self.assertEqual(features.shape[0], transformed_features.shape[0])
        self.assertEqual(features.shape[1], transformed_features.shape[1] - 4)


if __name__ == '__main__':
    unittest.main()

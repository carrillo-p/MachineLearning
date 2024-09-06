import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel

import sys
import os

# Añade el directorio src al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel

class TestLogisticModel(unittest.TestCase):
    def setUp(self):
        self.model = LogisticModel()
        # Crear datos de prueba
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def test_prepare_data(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.assertEqual(X_train.shape[0], 700)  # 70% para entrenamiento
        self.assertEqual(X_test.shape[0], 300)   # 30% para prueba
        self.assertEqual(y_train.shape[0], 700)
        self.assertEqual(y_test.shape[0], 300)

    def test_train(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        self.assertIsNotNone(self.model.model.coef_)

    def test_predict(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)
        self.assertEqual(predictions.shape[0], 300)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_evaluate(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        accuracy, conf_matrix, class_report = self.model.evaluate(X_test, y_test)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(conf_matrix, np.ndarray)
        self.assertIsInstance(class_report, str)

    def test_predict_proba(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        probabilities = self.model.predict_proba(X_test)
        self.assertEqual(probabilities.shape, (300, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_cross_validate(self):
        scores = self.model.cross_validate(self.X, self.y)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_roc_curve(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        fpr, tpr, roc_auc = self.model.roc_curve(X_test, y_test)
        self.assertIsInstance(fpr, np.ndarray)
        self.assertIsInstance(tpr, np.ndarray)
        self.assertIsInstance(roc_auc, float)

    def test_check_overfitting(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train)
        train_accuracy, test_accuracy = self.model.check_overfitting(X_train, y_train, X_test, y_test)
        self.assertIsInstance(train_accuracy, float)
        self.assertIsInstance(test_accuracy, float)
        self.assertTrue(0 <= train_accuracy <= 1)
        self.assertTrue(0 <= test_accuracy <= 1)

class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        self.model = XGBoostModel()
        # Crear datos de prueba
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def test_prepare_data(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.assertEqual(X_train.shape[0], 700)  # 70% para entrenamiento
        self.assertEqual(X_test.shape[0], 300)   # 30% para prueba
        self.assertEqual(y_train.shape[0], 700)
        self.assertEqual(y_test.shape[0], 300)

    def test_train(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        self.assertIsNotNone(self.model.model.get_booster())

    def test_predict(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        predictions = self.model.predict(X_test)
        self.assertEqual(predictions.shape[0], 300)
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_evaluate(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        accuracy, conf_matrix, class_report = self.model.evaluate(X_test, y_test)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(conf_matrix, np.ndarray)
        self.assertIsInstance(class_report, str)

    def test_predict_proba(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        probabilities = self.model.predict_proba(X_test)
        self.assertEqual(probabilities.shape, (300, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_cross_validate(self):
        scores = self.model.cross_validate(self.X, self.y)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_roc_curve(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        fpr, tpr, roc_auc = self.model.roc_curve(X_test, y_test)
        self.assertIsInstance(fpr, np.ndarray)
        self.assertIsInstance(tpr, np.ndarray)
        self.assertIsInstance(roc_auc, float)

    def test_check_overfitting(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        train_accuracy, test_accuracy = self.model.check_overfitting(X_train, y_train, X_test, y_test)
        self.assertIsInstance(train_accuracy, float)
        self.assertIsInstance(test_accuracy, float)
        self.assertTrue(0 <= train_accuracy <= 1)
        self.assertTrue(0 <= test_accuracy <= 1)

    def test_get_feature_importance(self):
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        self.model.train(X_train, y_train, X_test, y_test)
        feature_importance = self.model.get_feature_importance()
        self.assertEqual(len(feature_importance), 20)  # Número de características
        self.assertTrue(np.all(feature_importance >= 0))  # Todas las importancias deben ser no negativas

if __name__ == '__main__':
    unittest.main()
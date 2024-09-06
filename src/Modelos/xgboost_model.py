import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

class XGBoostModel:
    def __init__(self, eval_metric="logloss"):
        self.model = xgb.XGBClassifier(eval_metric=eval_metric)

    def prepare_data(self, X, y, test_size=0.3, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, X_train, y_train, X_test, y_test):
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, conf_matrix, class_report

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def cross_validate(self, X, y, cv=5):
        return cross_val_score(self.model, X, y, cv=cv)

    def roc_curve(self, X_test, y_test):
        y_prob = self.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def check_overfitting(self, X_train, y_train, X_test, y_test):
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, self.predict(X_test))
        return train_accuracy, test_accuracy

    def get_feature_importance(self):
        return self.model.feature_importances_
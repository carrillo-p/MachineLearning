import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

class LogisticModel:
    def __init__(self, max_iter=10000):
        self.model = LogisticRegression(max_iter=max_iter)
        self.scaler = StandardScaler()

    def prepare_data(self, X, y, test_size=0.3, random_state=42):
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        return accuracy, conf_matrix, class_report

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def cross_validate(self, X, y, cv=5):
        X_scaled = self.scaler.fit_transform(X)
        return cross_val_score(self.model, X_scaled, y, cv=cv)

    def roc_curve(self, X_test, y_test):
        y_probs = self.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

    def check_overfitting(self, X_train, y_train, X_test, y_test):
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, self.predict(X_test))
        return train_accuracy, test_accuracy
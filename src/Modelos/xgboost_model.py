import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib

class XGBoostModel:
    def __init__(self, eval_metric="logloss"):
        self.eval_metric = eval_metric
        self.model = None
        self.best_hyperparams = None

    def prepare_data(self, X, y, test_size=0.3, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def search_hyperparams(self, X_train, X_test, y_train, y_test):
        space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
        }
        
        def objective(space):
            clf = xgb.XGBClassifier(
                n_estimators=space['n_estimators'],
                max_depth=int(space['max_depth']),
                gamma=space['gamma'],
                reg_alpha=int(space['reg_alpha']),
                min_child_weight=int(space['min_child_weight']),
                colsample_bytree=space['colsample_bytree'],  # float entre 0 y 1
                eval_metric='logloss'  # Desde el constructor
            )
            
            evaluation = [(X_train, y_train), (X_test, y_test)]
            
            clf.fit(X_train, y_train,
                    eval_set=evaluation,
                    verbose=False)
            
            pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, pred > 0.5)
            print("SCORE:", accuracy)
            return {'loss': -accuracy, 'status': STATUS_OK}

        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                                space = space,
                                algo = tpe.suggest,
                                max_evals = 100,
                                trials = trials)
        
        self.best_hyperparams = best_hyperparams
        
    def init_model(self):
        self.model = xgb.XGBClassifier(
            n_estimators= 180,
            max_depth=int(self.best_hyperparams['max_depth']),
            gamma=self.best_hyperparams['gamma'],
            reg_alpha=int(self.best_hyperparams['reg_alpha']),
            reg_lambda=self.best_hyperparams['reg_lambda'],
            min_child_weight=int(self.best_hyperparams['min_child_weight']),
            colsample_bytree=self.best_hyperparams['colsample_bytree'],
            eval_metric=self.eval_metric
        )

    def train(self, X_train, y_train, X_test, y_test):
        self.search_hyperparams(X_train, X_test, y_train, y_test)
        self.init_model()
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
        if self.model is None:
            raise ValueError("Model not trained. Call 'train' before cross-validating.")
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

    def save_model(self, filename):
        joblib.dump(self.model, filename)


    @classmethod
    def load_model(cls, filename):
        instance = cls()
        instance.model = joblib.load(filename)
        return instance
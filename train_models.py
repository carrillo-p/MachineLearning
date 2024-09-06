import pandas as pd
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel

def train_and_save_models():
    # Cargar datos
    data = pd.read_csv('src/Data/airline_recoded.csv')
    X = data.drop('satisfaction', axis=1)
    y = data['satisfaction']

    # Entrenar y guardar modelo logístico
    logistic_model = LogisticModel()
    X_train, X_test, y_train, y_test = logistic_model.prepare_data(X, y)
    logistic_model.train(X_train, y_train)
    logistic_model.save_model('src/Modelos/logistic_model.joblib')

    # Entrenar y guardar modelo XGBoost
    xgboost_model = XGBoostModel()
    X_train, X_test, y_train, y_test = xgboost_model.prepare_data(X, y)
    xgboost_model.train(X_train, y_train, X_test, y_test)
    xgboost_model.save_model('src/Modelos/xgboost_model.joblib')

if __name__ == "__main__":
    train_and_save_models()
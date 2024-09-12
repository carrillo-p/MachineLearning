import pandas as pd
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from src.Modelos.stack_model import StackModel

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
    xgboost_model.search_hyperparams(X_train, X_test, y_train, y_test) # Buscamos los mejores hiperparametros
    xgboost_model.init_model()
    xgboost_model.train(X_train, y_train, X_test, y_test)
    xgboost_model.save_model('src/Modelos/xgboost_model.joblib')


    #Entrenar y guardar modelo Stack
    stack_model = StackModel()
    X_train, X_test, y_train, y_test = stack_model.prepare_data(X, y)
    stack_model.search_hyperparams(X_train, X_test, y_train, y_test)
    stack_model.init_model()
    stack_model.train(X_train, y_train, X_test, y_test)
    stack_model.save_model('src/Modelos/stack_model.joblib')

if __name__ == "__main__":
    train_and_save_models()
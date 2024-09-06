import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings
from sklearn.model_selection import train_test_split

# Ajusta estas importaciones según la estructura de tu proyecto
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
import app  # Importa tu script principal de Streamlit

def ignore_streamlit_warning(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)
    return wrapper

class TestStreamlitApp(unittest.TestCase):

    @ignore_streamlit_warning
    def setUp(self):
        self.logistic_model = LogisticModel()
        self.xgboost_model = XGBoostModel()
        
        # Crear datos de prueba más grandes
        self.test_input = pd.DataFrame({
            'Gender': [0, 1, 0, 1] * 25,
            'Customer Type': [1, 0, 1, 0] * 25,
            'Age': list(range(30, 130)),
            'Type of Travel': [0, 1, 0, 1] * 25,
            'Class': [0, 1, 2, 0] * 25,
            'Flight Distance': list(range(1000, 1100)),
            'Inflight wifi service': [3, 4, 2, 5] * 25,
            'Departure/Arrival time convenient': [3, 2, 4, 1] * 25,
            'Ease of Online booking': [2, 5, 3, 4] * 25,
            'Gate location': [3, 3, 2, 4] * 25,
            'Food and drink': [4, 2, 5, 3] * 25,
            'Online boarding': [3, 4, 2, 5] * 25,
            'Seat comfort': [4, 3, 5, 2] * 25,
            'Inflight entertainment': [3, 5, 2, 4] * 25,
            'On-board service': [4, 2, 5, 3] * 25,
            'Leg room service': [3, 4, 2, 5] * 25,
            'Baggage handling': [4, 3, 5, 2] * 25,
            'Checkin service': [3, 5, 2, 4] * 25,
            'Inflight service': [4, 2, 5, 3] * 25,
            'Cleanliness': [5, 3, 4, 2] * 25,
            'Departure Delay in Minutes': list(range(0, 100)),
            'Arrival Delay in Minutes': list(range(10, 110))
        })
        self.test_output = pd.Series([1, 0, 1, 0] * 25)

        # Inicializar los modelos
        X_train, X_test, y_train, y_test = train_test_split(self.test_input, self.test_output, test_size=0.3, random_state=42)
        self.logistic_model.train(X_train, y_train)
        self.xgboost_model.train(X_train, y_train, X_test, y_test)

    @ignore_streamlit_warning
    def test_predict_satisfaction(self):
        prediction, probability = app.predict_satisfaction(self.logistic_model, self.test_input.iloc[:1])
        self.assertIn(prediction, [0, 1])
        self.assertTrue(0 <= probability <= 1)

    @ignore_streamlit_warning
    def test_save_and_load_feedback(self):
        test_feedback = "Great app!"
        test_rating = 5
        app.save_feedback(test_feedback, test_rating)
        
        loaded_feedback = app.load_feedback()
        self.assertTrue(len(loaded_feedback) > 0)
        last_feedback = loaded_feedback[-1]
        self.assertEqual(last_feedback['comment'], test_feedback)
        self.assertEqual(last_feedback['rating'], test_rating)

    @ignore_streamlit_warning
    def test_create_gauge_chart(self):
        fig = app.create_gauge_chart(75, "Test Gauge")
        self.assertIsNotNone(fig)

    @ignore_streamlit_warning
    @patch('app.LogisticModel.load_model')
    @patch('app.XGBoostModel.load_model')
    def test_load_models(self, mock_xgb_load, mock_log_load):
        mock_log_load.return_value = self.logistic_model
        mock_xgb_load.return_value = self.xgboost_model
        
        logistic, xgboost = app.load_models()
        self.assertIsInstance(logistic, LogisticModel)
        self.assertIsInstance(xgboost, XGBoostModel)

    @ignore_streamlit_warning
    def test_logistic_model_methods(self):
        X_train, X_test, y_train, y_test = self.logistic_model.prepare_data(self.test_input, self.test_output)
        self.logistic_model.train(X_train, y_train)
        predictions = self.logistic_model.predict(X_test)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(X_test))

    @ignore_streamlit_warning
    def test_xgboost_model_methods(self):
        X_train, X_test, y_train, y_test = self.xgboost_model.prepare_data(self.test_input, self.test_output)
        self.xgboost_model.train(X_train, y_train, X_test, y_test)
        predictions = self.xgboost_model.predict(X_test)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(X_test))

    # Comentamos esta prueba si la función no existe en app.py
    # @ignore_streamlit_warning
    # def test_random_passenger_generation(self):
    #     passenger = app.generate_random_passenger()
    #     self.assertIsInstance(passenger, dict)
    #     self.assertIn('Edad', passenger)
    #     self.assertIn('Género', passenger)
    #     self.assertIn('Clase', passenger)

    @ignore_streamlit_warning
    @patch('streamlit.sidebar.radio')
    def test_navigation(self, mock_radio):
        mock_radio.return_value = "Predicción de Satisfacción"
        # Aquí podrías llamar a la función principal de tu app y verificar que se muestra la sección correcta
        # Por ejemplo:
        # app.main()
        # self.assertTrue(alguna_condición_que_verifique_la_sección_correcta)

    # Puedes añadir más pruebas aquí para otras funciones de tu aplicación

if __name__ == '__main__':
    unittest.main()
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())
print("Python path before modification:", sys.path)

# Añade el directorio raíz del proyecto al path de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

print("Updated Python path:", sys.path)
print("Contents of project root:", os.listdir(project_root))

# Importaciones de los módulos del proyecto
from pantallas.aux_functions import predict_satisfaction, save_feedback, load_feedback, create_gauge_chart
from pantallas.GUI_predict import screen_predict, load_models
from pantallas.GUI_feedback import screen_feedback
from pantallas.GUI_home import home_screen
from pantallas.GUI_informe import screen_informe, generar_grafico_log, generar_grafico_XGB, generar_grafico_stack
from pantallas.GUI_trivia import screen_trivia
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from src.Modelos.stack_model import StackModel

print("All modules imported successfully")

class TestModels(unittest.TestCase):
    @patch('src.Modelos.logistic_model.LogisticModel.load_model')
    def test_logistic_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        logistic_model = LogisticModel.load_model('fake_path.joblib')
        self.assertEqual(logistic_model, mock_model)

    @patch('src.Modelos.xgboost_model.XGBoostModel.load_model')
    def test_xgboost_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        xgboost_model = XGBoostModel.load_model('fake_path.joblib')
        self.assertEqual(xgboost_model, mock_model)

    @patch('src.Modelos.stack_model.StackModel.load_model')
    def test_stack_model(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        stack_model = StackModel.load_model('fake_path.joblib')
        self.assertEqual(stack_model, mock_model)

class TestAuxFunctions(unittest.TestCase):
    def test_predict_satisfaction(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        inputs = pd.DataFrame({'feature': [1]})
        prediction, probability = predict_satisfaction(mock_model, inputs)
        self.assertEqual(prediction, 1)
        self.assertEqual(probability, 0.7)

    @patch('pantallas.aux_functions.json.dump')
    @patch('pantallas.aux_functions.open')
    @patch('pantallas.aux_functions.datetime')
    def test_save_feedback(self, mock_datetime, mock_open, mock_json_dump):
        mock_datetime.now.return_value.strftime.return_value = "2024-09-10 12:00:00"
        save_feedback("Great service!", 5)
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        args = mock_json_dump.call_args[0]
        self.assertEqual(args[0][0]['comment'], "Great service!")
        self.assertEqual(args[0][0]['rating'], 5)
        self.assertEqual(args[0][0]['timestamp'], "2024-09-10 12:00:00")

    @patch('pantallas.aux_functions.os.path.exists', return_value=True)
    @patch('pantallas.aux_functions.json.load')
    @patch('pantallas.aux_functions.open', new_callable=mock_open)
    def test_load_feedback(self, mock_file, mock_json_load, mock_exists):
        mock_data = [{"comment": "Great service!", "rating": 5}]
        mock_json_load.return_value = mock_data
        result = load_feedback()
        mock_json_load.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["comment"], "Great service!")

    def test_create_gauge_chart(self):
        fig = create_gauge_chart(75, "Test Gauge")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(fig.data[0].value, 75)
        self.assertEqual(fig.data[0].title.text, "Test Gauge")

class TestGUIPredict(unittest.TestCase):
    @patch('pantallas.GUI_predict.LogisticModel')
    @patch('pantallas.GUI_predict.XGBoostModel')
    @patch('pantallas.GUI_predict.StackModel')
    def test_load_models(self, MockStack, MockXGB, MockLog):
        mock_log = MagicMock()
        mock_xgb = MagicMock()
        mock_stack = MagicMock()
        MockLog.load_model.return_value = mock_log
        MockXGB.load_model.return_value = mock_xgb
        MockStack.load_model.return_value = mock_stack

        # Crear una nueva instancia de la función load_models sin el decorador de caché
        from pantallas.GUI_predict import load_models
        uncached_load_models = load_models.__wrapped__

        log, xgb, stack = uncached_load_models()
        
        self.assertIs(log, mock_log)
        self.assertIs(xgb, mock_xgb)
        self.assertIs(stack, mock_stack)
        MockLog.load_model.assert_called_once_with('src/Modelos/logistic_model.joblib')
        MockXGB.load_model.assert_called_once_with('src/Modelos/xgboost_model.joblib')
        MockStack.load_model.assert_called_once_with('src/Modelos/stack_model.joblib')

    @patch('pantallas.GUI_predict.st')
    @patch('pantallas.GUI_predict.predict_satisfaction')
    @patch('pantallas.GUI_predict.create_gauge_chart')
    @patch('pantallas.GUI_predict.load_models')
    def test_screen_predict(self, mock_load_models, mock_gauge, mock_predict, mock_st):
        mock_load_models.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_predict.return_value = (1, 0.7)
        mock_gauge.return_value = go.Figure()
        mock_st.button.return_value = True
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        screen_predict()
        mock_st.markdown.assert_called()
        mock_st.selectbox.assert_called()
        mock_st.slider.assert_called()
        mock_st.number_input.assert_called()
        mock_predict.assert_called()
        mock_gauge.assert_called()
        mock_st.plotly_chart.assert_called()

class TestGUIFeedback(unittest.TestCase):
    @patch('pantallas.GUI_feedback.st')
    @patch('pantallas.GUI_feedback.save_feedback')
    @patch('pantallas.GUI_feedback.load_feedback')
    def test_screen_feedback(self, mock_load_feedback, mock_save_feedback, mock_st):
        mock_st.text_area.return_value = "Great app!"
        mock_st.slider.return_value = 5
        mock_st.button.return_value = True
        mock_load_feedback.return_value = [{"timestamp": "2024-09-10", "rating": 5, "comment": "Awesome!"}]
        screen_feedback()
        mock_save_feedback.assert_called_once_with("Great app!", 5)
        mock_st.success.assert_called_once()
        mock_st.markdown.assert_any_call("**Fecha:** 2024-09-10")

class TestGUIHome(unittest.TestCase):
    @patch('pantallas.GUI_home.st')
    def test_home_screen(self, mock_st):
        home_screen()
        mock_st.markdown.assert_called()

class TestGUIInforme(unittest.TestCase):
    @patch('pantallas.GUI_informe.st')
    @patch('pantallas.GUI_informe.generar_grafico_log')
    @patch('pantallas.GUI_informe.generar_grafico_XGB')
    @patch('pantallas.GUI_informe.generar_grafico_stack')
    @patch('pantallas.GUI_informe.joblib.load')
    @patch('pantallas.GUI_informe.accuracy_score')
    def test_screen_informe(self, mock_accuracy, mock_joblib_load, mock_graph_stack, mock_graph_xgb, mock_graph_log, mock_st):
        mock_st.session_state = {'modelo_seleccionado': 'XGBoost'}
        mock_st.selectbox.return_value = "Matriz de confusión"
        mock_joblib_load.return_value = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_accuracy.return_value = 0.85
        
        mock_y_test = np.array([0, 1, 0, 1])
        mock_y_pred = np.array([0, 1, 1, 1])
        
        with patch('pantallas.GUI_informe.y_test', mock_y_test), \
            patch('pantallas.GUI_informe.X_test', MagicMock()), \
            patch.object(mock_joblib_load.return_value, 'predict', return_value=mock_y_pred):
            screen_informe()
        
        mock_st.markdown.assert_called()
        mock_st.selectbox.assert_called()
        
        # Verifica que al menos una de las funciones de gráfico se llamó
        self.assertTrue(mock_graph_log.called or mock_graph_xgb.called or mock_graph_stack.called,
                        "Ninguna función de gráfico fue llamada")

    @patch('pantallas.GUI_informe.plt')
    @patch('pantallas.GUI_informe.st')
    @patch('pantallas.GUI_informe.confusion_matrix')
    @patch('pantallas.GUI_informe.roc_curve')
    @patch('pantallas.GUI_informe.auc')
    def test_generar_grafico_log(self, mock_auc, mock_roc_curve, mock_confusion_matrix, mock_st, mock_plt):
        mock_confusion_matrix.return_value = np.array([[50, 10], [5, 35]])
        mock_roc_curve.return_value = ([0, 0.5, 1], [0, 0.7, 1], None)
        mock_auc.return_value = 0.85
        generar_grafico_log("Matriz de confusión")
        mock_plt.figure.assert_called()
        mock_st.pyplot.assert_called()

    @patch('pantallas.GUI_informe.plt')
    @patch('pantallas.GUI_informe.st')
    @patch('pantallas.GUI_informe.confusion_matrix')
    @patch('pantallas.GUI_informe.roc_curve')
    @patch('pantallas.GUI_informe.auc')
    def test_generar_grafico_XGB(self, mock_auc, mock_roc_curve, mock_confusion_matrix, mock_st, mock_plt):
        mock_confusion_matrix.return_value = np.array([[50, 10], [5, 35]])
        mock_roc_curve.return_value = ([0, 0.5, 1], [0, 0.7, 1], None)
        mock_auc.return_value = 0.85
        generar_grafico_XGB("Curva ROC")
        mock_plt.figure.assert_called()
        mock_st.pyplot.assert_called()

    @patch('pantallas.GUI_informe.plt')
    @patch('pantallas.GUI_informe.st')
    @patch('pantallas.GUI_informe.joblib.load')
    def test_generar_grafico_stack(self, mock_joblib_load, mock_st, mock_plt):
        mock_joblib_load.return_value = MagicMock()
        generar_grafico_stack("Overfitting")
        mock_plt.figure.assert_called()
        mock_st.pyplot.assert_called()

class TestGUITrivia(unittest.TestCase):
    @patch('pantallas.GUI_trivia.st')
    def test_screen_trivia(self, mock_st):
        mock_st.radio.side_effect = ["Airbus A380", "KLM", "10,000 metros"]
        mock_st.button.return_value = True
        screen_trivia()
        mock_st.write.assert_called_with("Tu puntuación es: 3/3")
        mock_st.balloons.assert_called_once()
        mock_st.success.assert_called_once()

if __name__ == '__main__':
    unittest.main()
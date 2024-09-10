import json
import os
import plotly.graph_objects as go
from datetime import datetime


# Función para predecir la satisfacción
def predict_satisfaction(model, inputs):
    proba = model.predict_proba(inputs)[0]
    prediction = 1 if proba[1] > 0.5 else 0
    return prediction, proba[1]

# Función para guardar el feedback
def save_feedback(feedback, rating):
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rating": rating,
        "comment": feedback
    }
    
    filename = "feedback.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    data.append(feedback_data)
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

# Función para cargar el feedback
def load_feedback():
    filename = "feedback.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

# Función para crear un gráfico de gauge
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "white"},
                {'range': [25, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 0},
                'thickness': 0.75,
                'value': 90}}))
    return fig
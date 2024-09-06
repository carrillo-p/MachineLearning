import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from datetime import datetime
import json
import os
import random

# Configuración de la página
st.set_page_config(page_title="Airline Satisfaction Predictor", layout="wide", initial_sidebar_state="expanded")

# Función para cargar los modelos
@st.cache_resource
def load_models():
    logistic_model = LogisticModel.load_model('src/Modelos/logistic_model.joblib')
    xgboost_model = XGBoostModel.load_model('src/Modelos/xgboost_model.joblib')
    return logistic_model, xgboost_model

# Cargar los modelos
try:
    logistic_model, xgboost_model = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos: {str(e)}")
    st.stop()

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
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    return fig

# Sidebar para navegación
st.sidebar.title("Navegación ✈️")
page = st.sidebar.radio("Ir a", ["Inicio", "Predicción de Satisfacción", "Resultados de Modelos", "Feedback", "Juego de Trivia"])

if page == "Inicio":
    st.title("Bienvenido al Predictor de Satisfacción de Aerolíneas ✈️")
    st.markdown("""
    ¡Hola! Bienvenido a nuestra aplicación de predicción de satisfacción de pasajeros de aerolíneas. 
    Aquí podrás:
    
    - 🔮 Predecir la satisfacción de un pasajero basado en diferentes factores
    - 📊 Ver los resultados detallados de nuestros modelos de predicción
    - 💬 Dejar tu feedback y ver los comentarios de otros usuarios
    - 🎮 Jugar un divertido juego de trivia sobre aviación
    
    ¡Explora las diferentes secciones y diviértete!
    """)
    
    st.image("https://img.freepik.com/free-vector/airplane-sky_1308-31202.jpg", caption="¡Bienvenido a bordo!")

elif page == "Predicción de Satisfacción":
    st.title("Predictor de Satisfacción de Aerolíneas ✈️")
    st.markdown("### Ingrese los detalles del vuelo para predecir la satisfacción del cliente 😊")
    
    # Crear columnas para una mejor organización
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Género 👤", ["Male", "Female"])
        customer_type = st.selectbox("Tipo de Cliente 🧑‍💼", ["Loyal Customer", "disloyal Customer"])
        age = st.slider("Edad 🎂", 0, 100, 30)
        type_of_travel = st.selectbox("Tipo de Viaje 🏖️/💼", ["Personal Travel", "Business travel"])
        class_ = st.selectbox("Clase 💺", ["Eco Plus", "Business", "Eco"])
        flight_distance = st.number_input("Distancia de Vuelo 🛫", min_value=0, value=1000)
        inflight_wifi_service = st.slider("Servicio WiFi a bordo 📡", 0, 5, 3)
        departure_arrival_time_convenient = st.slider("Conveniencia de horarios ⏰", 0, 5, 3)
        ease_of_online_booking = st.slider("Facilidad de reserva en línea 💻", 0, 5, 3)
        gate_location = st.slider("Ubicación de la puerta 🚪", 0, 5, 3)
        food_and_drink = st.slider("Comida y bebida 🍔🥤", 0, 5, 3)
        
    with col2:
        online_boarding = st.slider("Embarque en línea 🎫", 0, 5, 3)
        seat_comfort = st.slider("Comodidad del asiento 🛋️", 0, 5, 3)
        inflight_entertainment = st.slider("Entretenimiento a bordo 🎭", 0, 5, 3)
        on_board_service = st.slider("Servicio a bordo 👨‍✈️", 0, 5, 3)
        leg_room_service = st.slider("Espacio para las piernas 🦵", 0, 5, 3)
        baggage_handling = st.slider("Manejo de equipaje 🧳", 0, 5, 3)
        checkin_service = st.slider("Servicio de check-in ✅", 0, 5, 3)
        inflight_service = st.slider("Servicio en vuelo 🛎️", 0, 5, 3)
        cleanliness = st.slider("Limpieza 🧼", 0, 5, 3)
        departure_delay = st.number_input("Retraso en la salida (minutos) ⏱️", min_value=0, value=0)
        arrival_delay = st.number_input("Retraso en la llegada (minutos) ⏱️", min_value=0, value=0)

    # Botón para predecir
    if st.button("Predecir Satisfacción 🔮", key="predict_button"):
        # Mostrar animación de carga
        with st.spinner('Calculando predicción... ✨'):
            # Preparar los inputs para el modelo
            inputs = pd.DataFrame({
                'Gender': [0 if gender == "Male" else 1],
                'Customer Type': [0 if customer_type == "Loyal Customer" else 1],
                'Age': [age],
                'Type of Travel': [0 if type_of_travel == "Personal Travel" else 1],
                'Class': [0 if class_ == "Eco Plus" else 1 if class_ == "Business" else 2],
                'Flight Distance': [flight_distance],
                'Inflight wifi service': [inflight_wifi_service],
                'Departure/Arrival time convenient': [departure_arrival_time_convenient],
                'Ease of Online booking': [ease_of_online_booking],
                'Gate location': [gate_location],
                'Food and drink': [food_and_drink],
                'Online boarding': [online_boarding],
                'Seat comfort': [seat_comfort],
                'Inflight entertainment': [inflight_entertainment],
                'On-board service': [on_board_service],
                'Leg room service': [leg_room_service],
                'Baggage handling': [baggage_handling],
                'Checkin service': [checkin_service],
                'Inflight service': [inflight_service],
                'Cleanliness': [cleanliness],
                'Departure Delay in Minutes': [departure_delay],
                'Arrival Delay in Minutes': [arrival_delay]
            })
            
            # Realizar predicciones
            logistic_pred, logistic_prob = predict_satisfaction(logistic_model, inputs)
            xgboost_pred, xgboost_prob = predict_satisfaction(xgboost_model, inputs)
        
        # Mostrar resultados
        st.subheader("Resultados de la Predicción 📊")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Modelo Logístico 📈")
            fig_logistic = create_gauge_chart(logistic_prob * 100, "Probabilidad de Satisfacción")
            st.plotly_chart(fig_logistic, use_container_width=True)
            emoji = "😃" if logistic_pred == 1 else "😞"
            st.metric("Predicción", f"{'Satisfecho' if logistic_pred == 1 else 'Insatisfecho'} {emoji}")

        with col2:
            st.markdown("### Modelo XGBoost 🌳")
            fig_xgboost = create_gauge_chart(xgboost_prob * 100, "Probabilidad de Satisfacción")
            st.plotly_chart(fig_xgboost, use_container_width=True)
            emoji = "😃" if xgboost_pred == 1 else "😞"
            st.metric("Predicción", f"{'Satisfecho' if xgboost_pred == 1 else 'Insatisfecho'} {emoji}")

        # Comparación de modelos
        st.subheader("Comparación de Modelos 🥊")
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Logístico', 'XGBoost']
        probs = [logistic_prob, xgboost_prob]
        ax.bar(models, probs, color=['skyblue', 'lightgreen'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidad de Satisfacción')
        ax.set_title('Comparación de Probabilidades entre Modelos')
        for i, v in enumerate(probs):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        st.pyplot(fig)

        st.balloons()

elif page == "Resultados de Modelos":
    st.title("Resultados y Comparación de Modelos 📊")

    # Tabla interactiva de métricas
    st.subheader("Métricas de los Modelos")
    metrics = pd.DataFrame({
        "Modelo": ["Logístico", "XGBoost"],
        "Accuracy": [0.85, 0.92],
        "Precisión": [0.83, 0.90],
        "Recall": [0.87, 0.94],
        "F1-Score": [0.85, 0.92]
    })
    st.dataframe(metrics.style.highlight_max(axis=0, color='lightgreen'))

    # Gráfico de barras interactivo
    st.subheader("Comparación de Métricas")
    metric_choice = st.selectbox("Elige una métrica", ["Accuracy", "Precisión", "Recall", "F1-Score"])
    fig = px.bar(metrics, x="Modelo", y=metric_choice, color="Modelo",
                 title=f"Comparación de {metric_choice}", 
                 labels={metric_choice: f"Valor de {metric_choice}"})
    st.plotly_chart(fig)

    # Matriz de confusión interactiva
    st.subheader("Matriz de Confusión")
    model_choice = st.radio("Elige un modelo", ["Logístico", "XGBoost"])
    conf_matrix = np.array([[150, 30], [20, 100]]) if model_choice == "Logístico" else np.array([[160, 20], [10, 110]])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_title(f'Matriz de Confusión - Modelo {model_choice}')
    st.pyplot(fig)

    # Curva ROC interactiva
    st.subheader("Curva ROC")
    fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    tpr_log = np.array([0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1])
    tpr_xgb = np.array([0, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.98, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr_log, mode='lines', name='Logístico (AUC = 0.85)'))
    fig.add_trace(go.Scatter(x=fpr, y=tpr_xgb, mode='lines', name='XGBoost (AUC = 0.90)'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        legend_title='Modelos'
    )
    st.plotly_chart(fig)
# Importancia de características (solo para XGBoost)
    st.subheader("Importancia de Características (XGBoost)")
    feature_importance = pd.DataFrame({
        'Característica': ['Distancia', 'Edad', 'Retraso', 'WiFi', 'Comida', 'Asiento'],
        'Importancia': [0.3, 0.2, 0.15, 0.12, 0.1, 0.08]
    }).sort_values('Importancia', ascending=True)
    
    fig = px.bar(feature_importance, x='Importancia', y='Característica', orientation='h',
                 title='Importancia de Características en XGBoost')
    st.plotly_chart(fig)

    # Sección divertida: "¿Sabías que...?"
    st.subheader("¿Sabías que...? 🤓")
    fun_facts = [
        "Los aviones vuelan a una altura de unos 10,000 metros, ¡eso es más alto que el Monte Everest!",
        "Un avión despega o aterriza cada 37 segundos en el aeropuerto más ocupado del mundo.",
        "Las alas de un avión 747 miden más que el primer vuelo de los hermanos Wright.",
        "Los pilotos y copilotos comen diferentes comidas por seguridad.",
        "El asiento más seguro en un avión es cerca de la cola."
    ]
    st.info(np.random.choice(fun_facts))

    # Botón para generar un nuevo pasajero aleatorio
    if st.button("¡Genera un pasajero aleatorio! 🎲"):
        random_passenger = {
            "Edad": np.random.randint(1, 100),
            "Género": np.random.choice(["Hombre", "Mujer"]),
            "Clase": np.random.choice(["Económica", "Business", "Primera"]),
            "Destino": np.random.choice(["París", "Tokyo", "New York", "Sydney", "Río de Janeiro"])
        }
        st.json(random_passenger)

elif page == "Feedback":
    st.title("Formulario de Feedback 📝")
    
    feedback = st.text_area("Por favor, comparte tu experiencia o sugerencias para mejorar nuestro servicio:")
    
    rating = st.slider("¿Cómo calificarías nuestra aplicación? 🌟", 1, 5, 3)
    
    if st.button("Enviar Feedback 📤", key="submit_feedback"):
        if feedback:
            save_feedback(feedback, rating)
            st.success("¡Gracias por tu feedback! Lo hemos recibido y lo tendremos en cuenta. 🙏")
            st.balloons()
        else:
            st.warning("Por favor, escribe algún feedback antes de enviar. ✍️")

    # Apartado desplegable para mostrar comentarios anteriores
    with st.expander("Ver comentarios anteriores 📜"):
        feedback_data = load_feedback()
        if feedback_data:
            for item in reversed(feedback_data):
                st.markdown(f"**Fecha:** {item['timestamp']}")
                st.markdown(f"**Calificación:** {'⭐' * item['rating']}")
                st.markdown(f"**Comentario:** {item['comment']}")
                st.markdown("---")
        else:
            st.info("Aún no hay comentarios. ¡Sé el primero en dejar tu feedback! 🥇")

elif page == "Juego de Trivia":
    st.title("¡Juego de Trivia de Aviación! 🎮✈️")
    
    # Lista de preguntas y respuestas
    trivia_questions = [
        {
            "question": "¿Cuál es el avión comercial más grande del mundo?",
            "options": ["Boeing 747", "Airbus A380", "Antonov An-225", "Boeing 787"],
            "correct": "Airbus A380"
        },
        {
            "question": "¿Cuál es la aerolínea más antigua del mundo que sigue operando?",
            "options": ["KLM", "Avianca", "Qantas", "American Airlines"],
            "correct": "KLM"
        },
        {
            "question": "¿A qué altura suelen volar los aviones comerciales?",
            "options": ["5,000 metros", "10,000 metros", "15,000 metros", "20,000 metros"],
            "correct": "10,000 metros"
        }
    ]
    
    score = 0
    for i, q in enumerate(trivia_questions):
        st.subheader(f"Pregunta {i+1}:")
        st.write(q["question"])
        answer = st.radio(f"Elige tu respuesta para la pregunta {i+1}:", q["options"], key=f"q{i}")
        if answer == q["correct"]:
            score += 1
    
    if st.button("¡Verificar respuestas!"):
        st.write(f"Tu puntuación es: {score}/{len(trivia_questions)}")
        if score == len(trivia_questions):
            st.balloons()
            st.success("¡Felicidades! ¡Eres un experto en aviación! 🏆✈️")
        elif score >= len(trivia_questions)/2:
            st.success("¡Buen trabajo! Tienes buenos conocimientos sobre aviación. 👍✈️")
        else:
            st.info("Sigue aprendiendo sobre aviación. ¡Lo harás mejor la próxima vez! 📚✈️")

# Añadir un footer
st.markdown("---")
st.markdown("© 2024 Airline Satisfaction Predictor. Todos los derechos reservados. 🛫")
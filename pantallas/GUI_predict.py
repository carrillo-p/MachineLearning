import streamlit as st
import pandas as pd
from pantallas.aux_functions import predict_satisfaction, create_gauge_chart
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from src.Modelos.stack_model import StackModel


@st.cache_resource
def load_models():
    logistic_model = LogisticModel.load_model('src/Modelos/logistic_model.joblib')
    xgboost_model = XGBoostModel.load_model('src/Modelos/xgboost_model.joblib')
    stack_model = StackModel.load_model('src/Modelos/stack_model.joblib')
    return logistic_model, xgboost_model, stack_model

try:
    logistic_model, xgboost_model, stack_model = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos: {str(e)}")
    st.stop()

def predict_satisfaction(model, inputs):
    proba = model.predict_proba(inputs)[0]
    prediction = 1 if proba[1] > 0.5 else 0
    return prediction, proba[1]

def screen_predict():
    st.markdown(f"""<h1 style="text-align: center;"> Predictor de Satisfacción </h1>""", unsafe_allow_html = True)
    st.markdown(f"""<h3 style="text-align: center;"> Ingrese los detalles del vuelo para predecir la satisfacción del cliente 😊</h3>""", unsafe_allow_html = True)

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
            stack_pred, stack_prob = predict_satisfaction(stack_model, inputs)
        
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
        
    
        st.markdown("<h3 style='text-align: center;'>Modelo Stacked</h3>", unsafe_allow_html=True)
        fig_stack = create_gauge_chart(stack_prob * 100, "Probabilidad de Satisfacción")
        st.plotly_chart(fig_stack, use_container_width=True)
        emoji = "😃" if stack_pred == 1 else "😞"
        st.metric("Predicción", f"{'Satisfecho' if stack_pred == 1 else 'Insatisfecho'} {emoji}")

        st.balloons()


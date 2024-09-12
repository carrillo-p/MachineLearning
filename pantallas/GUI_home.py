import streamlit as st
import pandas as pd

def home_screen():
    st.markdown(f"""<h1 style="text-align: center;"> Bienvenido al Predictor de Satisfacción de Aerolíneas ✈️</h1>""", unsafe_allow_html = True)
    st.markdown("""
    ¡Hola! Bienvenido a nuestra aplicación de predicción de satisfacción de pasajeros de aerolíneas. 
    Aquí podrás:
    
    - 🔮 Predecir la satisfacción de un pasajero basado en diferentes factores
    - 📊 Ver los resultados detallados de nuestros modelos de predicción
    - 💬 Dejar tu feedback y ver los comentarios de otros usuarios
    - 🎮 Jugar un divertido juego de trivia sobre aviación
    
    ¡Explora las diferentes secciones y diviértete!
    """)



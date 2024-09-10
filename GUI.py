import streamlit as st

from pantallas.GUI_home import home_screen
from pantallas.GUI_predict import screen_predict
from pantallas.GUI_informe import screen_informe
from pantallas.GUI_feedback import screen_feedback
from pantallas.GUI_trivia import screen_trivia

if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

def change_screen(new_screen):
    st.session_state.screen = new_screen

custom_html = """
<div class="banner">
    <img src="https://www.mtgpics.com/pics/art/10m/216_1.jpg" alt="Banner Image" height="250" width="1000">
</div>
<style>
    .banner {
        width: 100%;
        height: 700px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""

def apply_custom_css():
    st.markdown("""
    <style>
    /* Aplicar fondo al contenedor principal y cuerpo de la aplicación */
    .css-1v3fvcr {
        background-color: #f4f1de; /* Parchment-like background */
    }
    .css-1v0mbdj {
        background-color: #f4f1de; /* Parchment-like background */
    }
    body {
        background-color: #f4f1de; /* Parchment-like background */
        color: #FFFFFF; /* Dark brown text */
    }
    .stMarkdown {
        color: #FFFFFF; /* Dark brown text */
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .stButton button {
        width: 220px;
        height: 60px;
        background-color: #8B4513; /* Brown background */
        color: white; /* Text color */
        border: 1px solid #A0522D; /* Darker brown border color */
        border-radius: 8px;
        font-size: 18px; /* Font size */
        margin: 5px;
    }
    .stButton button:hover {
        background-color: #A0522D; /* Darker brown on hover */
    }
    .css-1v0mbdj a {
        color: #8B0000; /* Deep red link color */
    }
    .css-1v0mbdj a:hover {
        color: #B22222; /* Firebrick red on hover */
    }
    </style>
""", unsafe_allow_html=True)

st.components.v1.html(custom_html)
apply_custom_css()

st.sidebar.header("Menú de Navegación")
if st.sidebar.button("Home"):
    change_screen("home")
if st.sidebar.button("Predicción de Satisfacción"):
    change_screen("predict")
if st.sidebar.button("Informe de Modelos"):
    change_screen("informe")
if st.sidebar.button("Feedback"):
    change_screen("feedback")
if st.sidebar.button("Juego de trivia"):
    change_screen("trivia")

if st.session_state.screen == 'home':
    home_screen()
elif st.session_state.screen == 'predict':
    screen_predict()
elif st.session_state.screen == 'informe':
    screen_informe()
elif st.session_state.screen == 'feedback':
    screen_feedback()
elif st.session_state.screen == 'trivia':
    screen_trivia()


st.markdown("---")
st.markdown("© 2024 Airline Satisfaction Predictor. Todos los derechos reservados. 🛫")
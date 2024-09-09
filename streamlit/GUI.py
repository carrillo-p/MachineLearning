import streamlit as st
from GUI_screens import change_screen, home_screen, screen_predict

if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

st.set_page_config(layout='wide')

custom_html = """
<div class="banner">
    <img src="https://www.mtgpics.com/pics/art/sld/605.jpg" alt="Banner Image" height="500" width="1000">
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

st.components.v1.html(custom_html)

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
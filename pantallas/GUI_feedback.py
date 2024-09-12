import streamlit as st
from pantallas.aux_functions import save_feedback, load_feedback
import mysql.connector
from src.database.connection import create_connection, close_connection

# Función para guardar el feedback en la base de datos
def save_feedback(feedback, rating):
    connection = create_connection()
    cursor = connection.cursor()

    # Consulta para insertar los datos de feedback en la base de datos
    query = "INSERT INTO feedback (rating, comment) VALUES (%s, %s)"
    values = (rating, feedback)
    
    # Ejecutar la consulta y guardar los cambios
    cursor.execute(query, values)
    connection.commit()
    
    # Cerrar la conexión
    close_connection(connection)

# Función para cargar el feedback desde la base de datos
def load_feedback():
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    # Consulta para seleccionar todos los feedbacks en orden de fecha
    query = "SELECT rating, comment, timestamp FROM feedback ORDER BY timestamp DESC"
    
    # Ejecutar la consulta
    cursor.execute(query)
    feedback_data = cursor.fetchall()

    # Cerrar la conexión
    close_connection(connection)
    
    return feedback_data

def screen_feedback():
    st.markdown(f"""<h1 style="text-align: center;"> Formulario de Feedback 📝 </h1>""", unsafe_allow_html=True)

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
            for item in feedback_data:
                st.markdown(f"**Fecha:** {item['timestamp']}")
                st.markdown(f"**Calificación:** {'⭐' * item['rating']}")
                st.markdown(f"**Comentario:** {item['comment']}")
                st.markdown("---")
        else:
            st.info("Aún no hay comentarios. ¡Sé el primero en dejar tu feedback! 🥇")

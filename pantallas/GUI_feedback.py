import streamlit as st
from pantallas.aux_functions import save_feedback, load_feedback

def screen_feedback():
    st.markdown(f"""<h1 style="text-align: center;"> Formulario de Feedback 📝 </h1>""", unsafe_allow_html = True)

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


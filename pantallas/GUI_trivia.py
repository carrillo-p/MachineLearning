import streamlit as st

def screen_trivia():
    st.markdown(f"""<h1 style="text-align: center;"> ¡Juego de Trivia de Aviación! 🎮✈️ </h1>""", unsafe_allow_html = True)

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


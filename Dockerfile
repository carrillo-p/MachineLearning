# Etapa de construcción
FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa final
FROM python:3.12-slim

WORKDIR /app

# Copia las dependencias instaladas desde la etapa de construcción
COPY --from=builder /root/.local /root/.local

# Asegúrate de que los binarios instalados por pip estén en el PATH
ENV PATH=/root/.local/bin:$PATH

# Copia el código de la aplicación
COPY src/ /app/src/
COPY pantallas/ /app/pantallas/
COPY GUI.py /app/

# Expone el puerto en el que Streamlit se ejecutará
EXPOSE 8501

# Variables de entorno para Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "GUI.py"]

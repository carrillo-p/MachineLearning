# Etapa de construcción
FROM python:3.11-slim as builder

WORKDIR /app

# Copia los archivos de requerimientos
COPY requirements.txt .

# Instala las dependencias
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa final
FROM python:3.11-slim

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

# Variables de entorno para la conexión a la base de datos
ENV DB_HOST=db
ENV DB_USER=root
ENV DB_PASSWORD=example
ENV DB_NAME=airline_satisfaction

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "GUI.py"]
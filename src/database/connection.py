import mysql.connector
from dotenv import load_dotenv
import os

# Cargar variables de entorno del archivo .env si existe
load_dotenv()

def create_connection():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'db'),  # 'db' es el nombre del servicio en docker-compose
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', 'example'),
        database=os.getenv('DB_NAME', 'airline_satisfaction')
    )

def close_connection(connection):
    if connection.is_connected():
        connection.close()
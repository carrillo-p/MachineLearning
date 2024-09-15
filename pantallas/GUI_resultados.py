import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def screen_results():
    for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith("Clientes_simulados.csv")]:
                os.chdir(dirpath)

    clientes = pd.read_csv('Clientes_simulados.csv')

    columnas_feedback = ['Acierto_Logistico','Acierto_XGboost', 'Acierto_Stacked', 'Acierto_CNN']

    medias = clientes[columnas_feedback].mean()

    medias_porcentajes = medias * 100

    df_porcentajes = medias_porcentajes.reset_index()
    df_porcentajes.columns = ['Columna', 'Media (%)']

    top_3 = df_porcentajes.nlargest(3, 'Media (%)')

    diferencias = {
    (row1, row2): abs(top_3.loc[top_3['Columna'] == row1, 'Media (%)'].values[0] - top_3.loc[top_3['Columna'] == row2, 'Media (%)'].values[0])
    for i, row1 in enumerate(top_3['Columna'])
    for row2 in top_3['Columna'][i + 1:]
}

    mayor_diferencia = max(diferencias.values())
    if mayor_diferencia > 2:
        # Si hay una diferencia mayor al 2%, muestra la columna con el valor más alto
        columna_mas_alta = top_3.loc[top_3['Media (%)'].idxmax(), 'Columna']
        nombre_columna_mas_alta = columna_mas_alta.split('_')[-1]
        mensaje = f"El modelo recomendado es: {nombre_columna_mas_alta}"
    else:
        # Si ninguna diferencia es mayor al 2%, muestra 'XGBoost'
        mensaje = 'El modelo recomendado es: XGBoost'

    # Muestra el mensaje en un cuadro de texto en Streamlit
    st.title(mensaje)

    imagen_barra = Image.open('avion.png')

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (index, row) in enumerate(df_porcentajes.iterrows()):
        percentage = row['Media (%)']
        
        # Redimensionar la imagen en función del porcentaje
        imagen_redimensionada = imagen_barra.copy()
        ancho, alto = imagen_redimensionada.size
        nuevo_alto = int(alto * (percentage / 100))
        imagen_redimensionada = imagen_redimensionada.crop((0, alto - nuevo_alto, ancho, alto))
        
        # Mostrar la imagen en el gráfico
        ax_img = ax.imshow(imagen_redimensionada, aspect='auto', extent=(i - 0.4, i + 0.4, 0, percentage), zorder=1)
    
    ax.set_title('Media de columnas seleccionadas (en porcentajes)')
    ax.set_xlabel('Columnas')
    ax.set_ylabel('Media (%)')
    ax.set_xticks(range(len(df_porcentajes)))
    ax.set_xticklabels(df_porcentajes['Columna'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, (index, row) in enumerate(df_porcentajes.iterrows()):
        percentage = row['Media (%)']
        ax.text(i, percentage + 2, f'{percentage:.1f}', ha='center', va='bottom')


    st.pyplot(fig)

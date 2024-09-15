import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def screen_results():
    for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith("Clientes_simulados.csv")]:
                os.chdir(dirpath)

    clientes = pd.read_csv('Clientes_simulados.csv')

    columnas_feedback = ['Acierto_Logistico','Acierto_XGboost', 'Acierto_Stacked', 'Acierto_NeuralN']

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


    fig, ax = plt.subplots(figsize=(8, 6))
    bar_plot = sns.barplot(x='Columna', y='Media (%)', data=df_porcentajes, palette = 'viridis', ax=ax)
    ax.set_title('Media de columnas seleccionadas (en porcentajes)')
    ax.set_xlabel('Columnas')
    ax.set_ylabel('Media (%)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for p in bar_plot.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5, f'{height:.1f}', ha='center', va='bottom')


    st.pyplot(fig)

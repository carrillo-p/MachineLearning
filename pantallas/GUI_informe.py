import pandas as pd
import streamlit as st
import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from src.Modelos.stack_model import StackModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import numpy as np

def load_models():
    log_model = LogisticModel.load_model('src/Modelos/logistic_model.joblib')
    xgb_model = XGBoostModel.load_model('src/Modelos/xgboost_model.joblib')
    stack_model = StackModel.load_model('src/Modelos/stack_model.joblib')
    return log_model, xgb_model, stack_model

try:
    log_model, xgb_model, stack_model = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos: {str(e)}")
    st.stop()


for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if f.endswith("airline_recoded.csv")]:
                os.chdir(dirpath)

airline_df = pd.read_csv('airline_recoded.csv')


X = airline_df.drop(columns = ['satisfaction'])
y = airline_df['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def generar_grafico_log(tipo_grafico):

    y_pred = log_model.predict(X_test)
    y_prob = log_model.predict_proba(X_test)[:, 1]

    
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8,6))
    if tipo_grafico == "Matriz de confusión":
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Neutral o No satisfecho", "Satisfecho"], 
                    yticklabels=["Neutral o No Satisfecho", "Satisfecho"])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de confusión')
    elif tipo_grafico == "Curva ROC":
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr,tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    elif tipo_grafico == 'Variables más relevantes':
        coefficients = log_model.model.coef_[0]
        feature_names = X.columns

        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
            })
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title('Coeficientes de las Variables en el Modelo de Regresión Logística')
        plt.xlabel('Coeficiente')
        plt.ylabel('Variable')

    plt.tight_layout()  # Ajustar el diseño
    st.pyplot(plt.gcf())
            

def generar_grafico_XGB(tipo_grafico):
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8,6))
    if tipo_grafico == "Matriz de confusión":
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Neutral o No satisfecho", "Satisfecho"], 
                    yticklabels=["Neutral o No Satisfecho", "Satisfecho"])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de confusión')
        
    elif tipo_grafico == "Curva ROC":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
    elif tipo_grafico == "Overfitting":
        xgb_model_2 = joblib.load('../Modelos/xgboost_model.joblib')
        results = xgb_model_2.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
        plt.legend(loc='upper right')
        plt.xlabel('Number of Trees')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')

    elif tipo_grafico == 'Variables más relevantes':
        importance = xgb_model.model.feature_importances_
        feature_names = X.columns

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
            })
        
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Importancia de las Variables en el Modelo XGBoost')
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
    
    plt.tight_layout()  # Ajustar el diseño
    st.pyplot(plt.gcf())


def generar_grafico_stack(tipo_grafico):
    stack_model = joblib.load('../Modelos/stack_model.joblib')
    cv = 5
    train_accuracies = joblib.load('../Modelos/test_accuracies_kfold.pkl')
    test_accuracies = joblib.load('../Modelos/train_accuracies_kfold.pkl')

    y_pred = stack_model.predict(X_test)
    y_prob = stack_model.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(8,6))
    if tipo_grafico == "Matriz de confusión":
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Neutral o No satisfecho", "Satisfecho"], yticklabels=["Neutral o No Satisfecho", "Satisfecho"])
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de confusión')
        
    elif tipo_grafico == "Curva ROC":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
    elif tipo_grafico == "Overfitting":
        plt.plot(range(1, cv+1), train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(range(1, cv+1), test_accuracies, label='Test Accuracy', marker='o')

        plt.ylim(0.5, 1)
        plt.yticks(np.arange(0.50, 1.05, 0.05))

        plt.title('Train vs Test Accuracy en cada fold')
        plt.xlabel('Número de Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

    elif tipo_grafico == 'Variables más relevantes':
        xgb_importance = stack_model.named_estimators_['xgb'].feature_importances_
        rf_importance = stack_model.named_estimators_['rf'].feature_importances_
        feature_names = X.columns

        xgb_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb_importance,
        'Model': 'XGBoost'
        })
        rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance,
        'Model': 'RandomForest'
        })

        importance_df = pd.concat([xgb_importance_df, rf_importance_df])

        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', hue='Model', data=importance_df)
        plt.title('Importancia de las Variables en los Modelos Base del StackingClassifier')
        plt.xlabel('Importancia')
        plt.ylabel('Variable')
    
    plt.tight_layout()  # Ajustar el diseño
    st.pyplot(plt.gcf())

def screen_informe():

    if 'modelo_seleccionado' not in st.session_state:
        st.session_state['modelo_seleccionado'] = None

    st.markdown(f"""<h1 style="text-align: center;"> Información acerca de los modelos </h1>""", unsafe_allow_html = True)
    st.markdown(f"""<h3 style="text-align: center;"> Seleccione el modelo en el que está interesado para una explicación detallada. </h3>""", unsafe_allow_html = True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Modelo Logistico"):
            st.session_state['modelo_seleccionado'] = 'Logistic'
    with col2:
        if st.button("Modelo XGBoost"):
            st.session_state['modelo_seleccionado'] = 'XGBoost'
    with col3:
        if st.button("Modelo Stacked"):
            st.session_state['modelo_seleccionado'] = 'Stacked'

    if st.session_state['modelo_seleccionado'] == 'XGBoost':
        xgb_model = joblib.load('../Modelos/xgboost_model.joblib')

        y_pred = xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(xgb_model, X, y, cv=5)

        st.markdown("""
                    XGBoost es un algoritmo de _gradient boosting_ que ha ganado popularidad gracias a su velocidad y buenos resultados, especialmente para datos como los que estamos trabajando. 

                    XGBoost construye árboles de decisión de forma secuencial, donde cada árbol corrige errores generados en el árbol anterior. Incluye por defecto regularización (L1 y L2) para lidiar con problemas de overfitting, permitiendo además trabajar con otros hiperparámetros como la profundidad máxima (max_depth) o peso mínimo de nodos (min_child_weight).

                    Al trabajar de forma base con _gradient boosting_ es una forma muy sencilla de implementar técnicas de ensemble en un algoritmo de ML.

                    Para mejorar al máximo el rendimiento del modelo se han implementado los siguientes hiperparámetros:
                    - max_depth: Máxima profundidad del modelo ("ramas").
                    - gamma: Valor mínimo de pérdida para realizar una nueva partición del árbol. A mayor _gamma_ más conservador es el modelo.
                    - reg_alpha: Regularización L1, a mayor sea su valor más conservador es el modelo.
                    - min_child_weight: Suma mínima del peso en un "hijo". Si la partición del árbol resulta en una hoja nodo con una suma menor al valor especificado, el modelo dejará de realizar particiones.
                    - colsample_bytree: Submuestreo de columnas al construir el arbol. Especifica la fracción de columnas a submuestrear.
                    - eval_metric: La métrica para evaluar el modelo. Puede utilizarse varias, aquí usamos logloss, referida al valor negativo del log-likelihood.

                    Para determinar el mejor valor de los hiperparámetros se ha utilizado lo que se conoce como algoritmo de búsqueda ingenuo. Partimos de este algoritmo ya que no hay preconcepciones sobre el modelo ni sus hiperparámetros, por lo que es el mejor punto de partida para establecer una linea base.

                    En concreto se ha optado por realizar una búsqueda en cuadrícula (_grid search_), en la que definimos un espacio de búsqueda para los hiperparámetros y probamos las combinaciones para encontrar la mejor configuración.

                    Para implementar esta metodología se ha empleado la libreria _hyperopt_.

                    Se ha utilizado también validación cruzada para asegurar en la medida de lo posible que el modelo no presenta overfitting, con 5 muestras cruzadas.

                    ### Evaluación del modelo

                    A continuación pueden revisarse las diferentes gráficas que suelen emplearse para determinar el ajuste del modelo.
                    """)
        graph = st.selectbox("Gráficas", options = ["Matriz de confusión", "Curva ROC", "Overfitting", 'Variables más relevantes'])
        if graph == "Matriz de confusión":
             generar_grafico_XGB(graph)
             st.markdown(f"""
                        #### Matriz de confusión

                        La matriz de confusión nos permite evaluar el número de errores que comete el modelo en sus predicciones con el conjunto de prueba. Podemos dividir las predicciones en cuatro categorías:
                        - True Positives: En nuestro caso cuando el modelo acierta que el cliente quedó satisfecho. Cuadrante inferior derecho.
                        - True Negatives: En nuestro caso cuando el modelo acierta que el cliente quedó insatisfecho. Cuadrante superior izquierdo.
                        - False Positive: El modelo predice un cliente satisfecho cuando está insatisfecho. Cuadrante superior derecho.
                        - False Negatives: El modelo predice un cliente insatisfecho cuando está satisfecho. Cuadrante inferior izquierdo.

                        Como puede verse en la gráfica, el modelo presenta un número muy alto de TP y TN, lo cual indica un buen ajuste del modelo. 

                        Podemos operativizarlo de manera numérica calculando la precisión, calculada con estos valores numéricos.

                        En nuestro caso encontramos un ratio de precisión total del {np.round(accuracy, 2)}.

                        Podemos analizar también los valores del reporte de clasificación, donde se incluyen las medidas de exhaustividad (_recall_), proporción de verdaderos positivos entre los casos positivos (TP+NP), y el F1-score, media armónica de la precisión y la exhaustividad.

                        Reporte de clasificación: 

                        - Recall: Neutral o no satisfecho = 0.98 / Satisfecho = 0.94
                        - F1-score: Neutral o no satisfecho = 0.97 / Satisfecho = 0.96

                        Todos los valores obtenidos muestran valores por encima del 0.94, demostrando un gran ajuste del modelo.""")
        if graph == "Curva ROC":
             generar_grafico_XGB(graph)
             st.markdown(f"""
                        #### Curva ROC

                        La curva ROC representa la compensación entre la tasa de True Positives y la tasa de False Positives. También puede conceptualizarse como una gráfica que muestra el poder estadístico como función del error tipo I.

                        Cuanto más se aproxime la curva a la esquina superior izquierda del gráfico, mejor consideraremos el modelo, ya que tiene una buena tasa de true positives.

                        Además de la gráfica, tenemos el valor AUC (_area under the curve_) como una cuantificación del rendimiento del modelo, basada en el área debajo de la curva. Este área representa la capacidad del modelo para distinguir entre positivos y negativos. A mayor valor, mayor capacidad de discriminación (del 0 al 1).

                        En el caso que nos atañe, ambos conceptos nos sirven para evaluar cómo de bien nuestro modelo es capaz de detectar la satisfacción de los clientes. 

                        En este caso atendiendo tanto a la gráfica, como al alto valor AUC (0.99). Podemos concluir que el modelo está discriminando de forma muy eficiente los casos de satisfacción.
                         """)
        if graph == "Overfitting":
             generar_grafico_XGB(graph)
             st.markdown(f"""
                        #### Sobreajuste (Overfitting)

                        Por último, evaluamos el sobreajuste del modelo. El sobreajuste hace referencia al fenómeno por el cual un modelo se acostumbra demasiado a los datos de entrenamiento y no es capaz de generalizar el entrenamiento a nuevos datos de prueba.

                        Como puede verse en la gráfica de sobreajuste, para este modelo la tasa de acierto para el conjunto de entrenamiento y de prueba es muy similar, y, si calculamos el valor concreto del sobreajuste, encontramos que no llega a un 5% que es lo solicitado por el cliente.

                        Aún así, para aumentar la confianza en que el modelo no sobreajuste, se ha implementado validación cruzada, encontrándose el mismo resultado.

                        - Validación cruzada: {np.round(cv_scores, 2)}
                        - Media de las puntuaciones: {np.round(cv_scores.mean(), 2)}
                        """)
        if graph == 'Variables más relevantes':
            generar_grafico_XGB(graph)
            st.markdown(f"""
                    Este gráfico representa las variables que más peso tienen a la hora de determinar la probabilidad de pertenecer a la categoría de "satisfecho".
                        """)

    if st.session_state['modelo_seleccionado'] == 'Logistic':
            y_pred = log_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown("""
                        La regresión logística es un tipo es un tipo de modelo líneal que analiza la relación entre una variable dependiente binaria (0 - 1) y una  o más variables independientes (las cuales pueden ser de diferentes tipos). 

                        Este tipo de modelo extrae los coeficientes de regresión de las variables independientes para predecir la probabilidad (en _odds_ o en Probabilidad) de pertenecer a la categoría 1 (en este caso la probabilidad de estar satisfecho con el vuelo). Por tanto es un buen modelo para trabajar con machine learning en problemas de clasificación binaria con aprendizaje supervisado.

                        Su principal ventaja es que es un modelo facil de implementar y de interpretar, en especial en machine learning donde no tenemos que trabajar con métricas logit. Es especialmente relevante en conjuntos de datos que son linealmente separables, además, permite ver el peso de las diferentes variables gracias a sus coeficientes de regresión, permitiendo además ver su dirección (si aumentan o disminuyen la probabilidad de que el cliente esté satisfecho).

                        Permite la inclusión de hiperparámteros para mejorar su rendimiento. En este caso se ha optado por utilizar regularización, el cual penaliza modelos complejos para evitar que se de sobreajuste en el modelo.

                        Se ha utilizado también validación cruzada para asegurar en la medida de lo posible que el modelo no presenta overfitting, con 5 muestras cruzadas.

                        ### Evaluación del modelo

                        A continuación pueden revisarse las diferentes gráficas que suelen emplearse para determinar el ajuste del modelo.

                        """)
            graph = st.selectbox("Gráficas", options = ["Matriz de confusión", "Curva ROC", "Overfitting", 'Variables más relevantes'])
            if graph == "Matriz de confusión":
                generar_grafico_log(graph)
                st.markdown(f"""
                        #### Matriz de confusión

                        La matriz de confusión nos permite evaluar el número de errores que comete el modelo en sus predicciones con el conjunto de prueba. Podemos dividir las predicciones en cuatro categorías:
                        - True Positives: En nuestro caso cuando el modelo acierta que el cliente quedó satisfecho. Cuadrante inferior derecho.
                        - True Negatives: En nuestro caso cuando el modelo acierta que el cliente quedó insatisfecho. Cuadrante superior izquierdo.
                        - False Positive: El modelo predice un cliente satisfecho cuando está insatisfecho. Cuadrante superior derecho.
                        - False Negatives: El modelo predice un cliente insatisfecho cuando está satisfecho. Cuadrante inferior izquierdo.

                        Como puede verse en la gráfica, el modelo presenta un número alto de TP y TN, lo cual indica un buen ajuste del modelo. 

                        Podemos operativizarlo de manera numérica calculando la precisión, calculada con estos valores numéricos.

                        En nuestro caso encontramos un ratio de precisión total del {np.round(accuracy, 2)}.

                        Podemos analizar también los valores del reporte de clasificación, donde se incluyen las medidas de exhaustividad (_recall_), proporción de verdaderos positivos entre los casos positivos (TP+NP), y el F1-score, media armónica de la precisión y la exhaustividad.

                        Reporte de clasificación: 

                        - Recall: Neutral o no satisfecho = 0.91 / Satisfecho = 0.83
                        - F1-score: Neutral o no satisfecho = 0.89 / Satisfecho = 0.85

                        Todos los valores obtenidos muestran valores por encima del 0.80, demostrando un buen ajuste del modelo.""")
            if graph == 'Curva ROC':
                generar_grafico_log(graph)
                st.markdown(f"""
                        #### Curva ROC

                        La curva ROC representa la compensación entre la tasa de True Positives y la tasa de False Positives. También puede conceptualizarse como una gráfica que muestra el poder estadístico como función del error tipo I.

                        Cuanto más se aproxime la curva a la esquina superior izquierda del gráfico, mejor consideraremos el modelo, ya que tiene una buena tasa de true positives.

                        Además de la gráfica, tenemos el valor AUC (_area under the curve_) como una cuantificación del rendimiento del modelo, basada en el área debajo de la curva. Este área representa la capacidad del modelo para distinguir entre positivos y negativos. A mayor valor, mayor capacidad de discriminación (del 0 al 1).

                        En el caso que nos atañe, ambos conceptos nos sirven para evaluar cómo de bien nuestro modelo es capaz de detectar la satisfacción de los clientes. 

                        En este caso atendiendo tanto a la gráfica, como al alto valor AUC (0.92). Podemos concluir que el modelo está discriminando de forma muy eficiente los casos de satisfacción.
                            """)
            if graph == "Overfitting":
                st.markdown(f"""
                        #### Sobreajuste (Overfitting)

                        Por último, evaluamos el sobreajuste del modelo. El sobreajuste hace referencia al fenómeno por el cual un modelo se acostumbra demasiado a los datos de entrenamiento y no es capaz de generalizar el entrenamiento a nuevos datos de prueba.
                            
                        Si calculamos el valor concreto del sobreajuste, encontramos que no llega a un 5% que es lo solicitado por el cliente. Se muestra que un mayor valor en la regularización mejora el acierto pero no modifica significativamente el sobreajuste del modelo.
                        """)
            if graph == 'Variables más relevantes':
                 generar_grafico_log(graph)
                 st.markdown(f"""
                            Este gráfico representa las variables que más peso tienen a la hora de determinar la probabilidad de pertenecer a la categoría de "satisfecho". Los coeficientes de regresión negativos indican aquellas categorías que reducen la probabilidad y los positivos las que la aumentan.
                             """)
            
    if st.session_state['modelo_seleccionado'] == 'Stacked':
            stack_model = joblib.load('../Modelos/stack_model.joblib')
            y_pred = stack_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.markdown(f"""
                    Este modelo ha sido generado utilizando la técnica de ensemble de Stacking.

                    Esta técnica combina diferentes tipos de algoritmos de ML con el objetivo de conseguir resultados superiores a los que podría obtener un único algoritmo.

                    A pesar de que ya hemos utilizado una técnica de ensemble con el _gradient boosting_, hemos querido demostrar el funcionamiento de un modelo que lo combine con _stacking_.

                    Por ello, se ha decidido utilizar el modelo XGBoost que hemos podido ver que ofrece unos buenos resultados, en combinación con un modelo de regresión logística y un modelo de _random forest_ para comprobar si es posible mejorar el modelo.

                    Para mejorar al máximo el rendimiento del modelo se han implementado los mismos hiperparámetros que en el modelo XGBoost:
                    - max_depth: Máxima profundidad del modelo ("ramas").
                    - gamma: Valor mínimo de pérdida para realizar una nueva partición del árbol. A mayor _gamma_ más conservador es el modelo.
                    - reg_alpha: Regularización L1, a mayor sea su valor más conservador es el modelo.
                    - min_child_weight: Suma mínima del peso en un "hijo". Si la partición del árbol resulta en una hoja nodo con una suma menor al valor especificado, el modelo dejará de realizar particiones.
                    - colsample_bytree: Submuestreo de columnas al construir el arbol. Especifica la fracción de columnas a submuestrear.
                    - eval_metric: La métrica para evaluar el modelo. Puede utilizarse varias, aquí usamos logloss, referida al valor negativo del log-likelihood.

                    Para determinar el mejor valor de los hiperparámetros se ha utilizado el mismo proceso que en el modelo con XGBoost, con algoritmo de búsqueda ingenuo. 

                    En concreto se ha optado por realizar una búsqueda en cuadrícula (_grid search_), en la que definimos un espacio de búsqueda para los hiperparámetros y probamos las combinaciones para encontrar la mejor configuración. Utilzando _hyperopt_.


                    Se ha utilizado también validación cruzada para asegurar en la medida de lo posible que el modelo no presenta overfitting, con 5 muestras cruzadas.

                    ## Evaluación del modelo

                    A continuación pueden revisarse las diferentes gráficas que suelen emplearse para determinar el ajuste del modelo.
                    """)
            graph = st.selectbox("Gráficas", options = ["Matriz de confusión", "Curva ROC", "Overfitting", 'Variables más relevantes'])
            if graph == "Matriz de confusión":
                generar_grafico_stack(graph)
                st.markdown(f"""
                        #### Matriz de confusión

                        La matriz de confusión nos permite evaluar el número de errores que comete el modelo en sus predicciones con el conjunto de prueba. Podemos dividir las predicciones en cuatro categorías:
                        - True Positives: En nuestro caso cuando el modelo acierta que el cliente quedó satisfecho. Cuadrante inferior derecho.
                        - True Negatives: En nuestro caso cuando el modelo acierta que el cliente quedó insatisfecho. Cuadrante superior izquierdo.
                        - False Positive: El modelo predice un cliente satisfecho cuando está insatisfecho. Cuadrante superior derecho.
                        - False Negatives: El modelo predice un cliente insatisfecho cuando está satisfecho. Cuadrante inferior izquierdo.

                        Como puede verse en la gráfica, el modelo presenta un número muy alto de TP y TN, lo cual indica un buen ajuste del modelo. 

                        Podemos operativizarlo de manera numérica calculando la precisión, calculada con estos valores numéricos.

                        En nuestro caso encontramos un ratio de precisión total del {np.round(accuracy, 2)}.

                        Podemos analizar también los valores del reporte de clasificación, donde se incluyen las medidas de exhaustividad (_recall_), proporción de verdaderos positivos entre los casos positivos (TP+NP), y el F1-score, media armónica de la precisión y la exhaustividad.

                        Reporte de clasificación: 

                        - Recall: Neutral o no satisfecho = 0.98 / Satisfecho = 0.94
                        - F1-score: Neutral o no satisfecho = 0.97 / Satisfecho = 0.96

                        Todos los valores obtenidos muestran valores por encima del 0.94, demostrando un gran ajuste del modelo.""")
            if graph == "Curva ROC":
                generar_grafico_stack(graph)
                st.markdown(f"""
                        #### Curva ROC

                        La curva ROC representa la compensación entre la tasa de True Positives y la tasa de False Positives. También puede conceptualizarse como una gráfica que muestra el poder estadístico como función del error tipo I.

                        Cuanto más se aproxime la curva a la esquina superior izquierda del gráfico, mejor consideraremos el modelo, ya que tiene una buena tasa de true positives.

                        Además de la gráfica, tenemos el valor AUC (_area under the curve_) como una cuantificación del rendimiento del modelo, basada en el área debajo de la curva. Este área representa la capacidad del modelo para distinguir entre positivos y negativos. A mayor valor, mayor capacidad de discriminación (del 0 al 1).

                        En el caso que nos atañe, ambos conceptos nos sirven para evaluar cómo de bien nuestro modelo es capaz de detectar la satisfacción de los clientes. 

                        En este caso atendiendo tanto a la gráfica, como al alto valor AUC (0.99). Podemos concluir que el modelo está discriminando de forma muy eficiente los casos de satisfacción.
                         """)
            if graph == "Overfitting":
                generar_grafico_stack(graph)
                st.markdown(f"""
                        #### Sobreajuste (Overfitting)

                        Por último, evaluamos el sobreajuste del modelo. El sobreajuste hace referencia al fenómeno por el cual un modelo se acostumbra demasiado a los datos de entrenamiento y no es capaz de generalizar el entrenamiento a nuevos datos de prueba.

                        Como puede verse en la gráfica de sobreajuste, para este modelo la tasa de acierto para el conjunto de entrenamiento y de prueba es muy similar, y, si calculamos el valor concreto del sobreajuste, encontramos que no llega a un 5% que es lo solicitado por el cliente. Además se muestra que se repite para los 5 conjuntos de validación cruzada.
                        """)
            if graph == 'Variables más relevantes':
                generar_grafico_stack(graph)
                st.markdown(f"""
                            Este gráfico representa las variables que más peso tienen a la hora de determinar la probabilidad de pertenecer a la categoría de "satisfecho". 
                            Al tratarsse de un modelo configurado mediante la técnica de stacking, no podemos obtener una única medida de la importancia de cada variable. Por tanto, mostramos en la gráfica los índices para dos de los modelos que configuran el stack, random forest y XGBoost. Se puede observar que muchas de las variables tienen el mismo orden en importancia, pero en algunos casos los modelos difieren en dicha medida.
                            
                            De esta forma puede verse como opera un modelo de Stack, combinando ambos modelos con sus diferencias para generar la mejor predicción posible con los datos disponibles.
                            """)
            


import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith("airline_recoded.csv")]: # usando un list comprehension para filtrar los archivos que terminan en funciones_aux.py
            os.chdir(dirpath)

airline_df = pd.read_csv('airline_recoded.csv')

X = airline_df.drop(columns=['satisfaction'])
y = airline_df['satisfaction']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, '../Modelos/scaler.save')


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()

# Añadir capas
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],),  kernel_regularizer=l2(0.0001)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Clasificación binaria

adam = Adam(learning_rate=0.000098) #0.000098

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_split = 0.2)

model.save('../Modelos/neuronal.keras')

import pickle
with open('../Modelos/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
import tensorflow as tf
import numpy as np

# Datos de entrenamiento
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Construir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'),  # Capa de entrada con 2 neuronas y función de activación sigmoidal
    tf.keras.layers.Dense(1, activation='sigmoid')  # Capa de salida con 1 neurona y función de activación sigmoidal
])

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam')  # Función de costo: error cuadrático medio, Optimizador: Adam

# Entrenar el modelo
model.fit(x_train, y_train, epochs=1000, verbose=0)

# Predecir nuevos datos
x_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(x_new)

# Imprimir las predicciones
for i, x in enumerate(x_new):
    print(f'Entrada: {x} - Predicción: {predictions[i]}')

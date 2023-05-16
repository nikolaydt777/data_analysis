from sklearn.metrics import confusion_matrix
import numpy as np

def polinomial_prediction(titanic_data, polynomial_coefficients, threshold):
    # Crear una matriz con las variables predictoras y la variable a predecir
    X = titanic_data.drop('Survived', axis=1)
    y = titanic_data['Survived']
    # Transformar las variables predictoras en un array de numpy para poder hacer el producto con los coeficientes del polinomio
    X_array = np.array(X)
    # Calcular la suma ponderada de las variables predictoras según los coeficientes del polinomio
    prediction = np.dot(X_array, polynomial_coefficients)
    # Clasificar a los pasajeros como sobrevivientes o no sobrevivientes según el umbral dado
    prediction_classes = np.where(prediction > threshold, 1, 0)
    # Calcular la matriz de confusión
    cm = confusion_matrix(y, prediction_classes)
    # Calcular la precisión del modelo
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    return cm, accuracy

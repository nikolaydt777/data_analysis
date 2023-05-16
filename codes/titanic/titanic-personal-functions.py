import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def custom_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def custom_logistic_regression(X, y, alpha, num_iterations):
    m, n = X.shape
    weights = np.zeros((n, 1))
    for i in range(num_iterations):
        z = np.dot(X, weights)
        h = custom_sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        weights -= alpha * gradient
    return weights

# Lee los datos del archivo CSV
df = pd.read_csv('titanic.csv')

# Realiza una limpieza básica de datos
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # Elimina columnas no importantes
df = df.dropna()  # Elimina filas con valores perdidos

# Crea variables dummy para las columnas categóricas
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Divide los datos en entrenamiento y prueba
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Entrena un modelo de regresión logística personalizado
alpha = 0.01
num_iterations = 1000
weights = custom_logistic_regression(X_train, y_train, alpha, num_iterations)

# Realiza predicciones en el conjunto de prueba
y_pred = np.round(custom_sigmoid(np.dot(X_test, weights)))

# Calcula la matriz de confusión y el porcentaje de precisión
cm = confusion_matrix(y_test, y_pred)
accuracy = np.trace(cm) / np.sum(cm)

# Imprime los resultados
print('Matriz de confusión:')
print(cm)
print('Porcentaje de precisión:', accuracy)

# Grafica los datos y la recta de predicción
import matplotlib.pyplot as plt

x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.round(custom_sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights)))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Predicciones de supervivencia del Titanic')
plt.show()

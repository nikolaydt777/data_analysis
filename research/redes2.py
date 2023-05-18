import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Carga de datos
data = pd.read_csv('titanic.csv')

# Análisis estadístico de los datos
print("Análisis estadístico:")
print(data.describe())

# Preprocesamiento de los datos
data = data.dropna()  # Eliminar filas con valores faltantes
X = data[['Age', 'Fare', 'Pclass']]  # Variables predictoras
y = data['Survived']  # Variable objetivo

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento de la red neuronal
model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test)

# Precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("\nPrecisión del modelo:", accuracy)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(cm)

# Visualización de la matriz de confusión
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ['No sobrevivió', 'Sobrevivió'])
plt.yticks(tick_marks, ['No sobrevivió', 'Sobrevivió'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

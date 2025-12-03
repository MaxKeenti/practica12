# Importar las bibliotecas necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de digitos
digits = datasets.load_digits()

# Mostrar digitos en forma de arreglos (opcional descomentar para mirarlos)
# print(digits)

# Dividir los datos en características (X) y etiquetas (y)
X = digits.images
y = digits.target

# Visualizar una muestra de las imágenes y etiquetas
n_muestras = 5
plt.figure(figsize=(10, 2))
for i in range(n_muestras):
    plt.subplot(1, n_muestras, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f"Digito {y[i]}")
    plt.axis('off')
plt.show()

# Preprocesar las imágenes a formato 1D
X = X.reshape([X.shape[0],-1])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Crear un modelo de SVM con el kernel lineal
modelo = SVC(kernel='linear')

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de pruebas
y_pred = modelo.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM con kernellineal: {precision:.2f}")

# Mostrar un informe de clasificación
print("Informe de clasificación")
print(classification_report(y_test, y_pred))
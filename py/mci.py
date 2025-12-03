# Importar las bibliotecas necesarias
# datasets: contiene conjuntos de datos de prueba, incluyendo el de dígitos
# train_test_split: función para dividir los datos en conjuntos de entrenamiento y prueba
# SVC: Support Vector Classifier, el algoritmo de clasificación que usaremos
# classification_report, accuracy_score: métricas para evaluar el rendimiento del modelo
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# Cargar el conjunto de datos de digitos
# Este dataset contiene imágenes de 8x8 píxeles de dígitos escritos a mano
digits = datasets.load_digits()

# Mostrar digitos en forma de arreglos (opcional descomentar para mirarlos)
# print(digits)

# Dividir los datos en características (X) y etiquetas (y)
# X contiene las matrices de imágenes (los datos de entrada)
# y contiene los números reales que representan cada imagen (las etiquetas)
X = digits.images
y = digits.target

# Visualizar una muestra de las imágenes y etiquetas
# Se mostrarán los primeros 5 dígitos del dataset para entender con qué estamos trabajando
n_muestras = 5
plt.figure(figsize=(10, 2))
for i in range(n_muestras):
    plt.subplot(1, n_muestras, i + 1)
    # Usamos 'plasma' para una mejor visualización de la intensidad de los píxeles
    plt.imshow(X[i], cmap='plasma') 
    plt.title(f"Digito {y[i]}")
    plt.axis('off')

# Guardar la figura generada en la carpeta docs/media
# Esto permite incluirla automáticamente en el reporte
output_dir = "../docs/media"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, "figure_1.png"))
print(f"Figura guardada en {os.path.join(output_dir, 'figure_1.png')}")
# plt.show() # Comentado para evitar bloqueo en ejecución automática

# Preprocesar las imágenes a formato 1D
# Las imágenes son matrices 2D (8x8), pero el modelo SVM requiere un vector 1D (64 elementos)
# reshape transforma cada imagen de 8x8 a un vector de 64
X = X.reshape([X.shape[0],-1])

# Dividir los datos en conjuntos de entrenamiento y prueba
# Usamos el 20% de los datos para prueba (test_size=0.2) y el 80% para entrenamiento
# random_state=43 asegura que la división sea reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Crear un modelo de SVM con el kernel lineal
# El kernel lineal es adecuado cuando los datos son linealmente separables
modelo = SVC(kernel='linear')

# Entrenar el modelo con los datos de entrenamiento
# El modelo aprende la relación entre los píxeles (X_train) y los dígitos (y_train)
modelo.fit(X_train, y_train)

# Predecir las etiquetas para el conjunto de pruebas
# Usamos el modelo entrenado para predecir los dígitos de las imágenes que no ha visto (X_test)
y_pred = modelo.predict(X_test)

# Calcular la precisión del modelo
# Comparamos las predicciones (y_pred) con los valores reales (y_test)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM con kernel lineal: {precision:.2f}")

# Mostrar un informe de clasificación detallado
# Incluye precision, recall, f1-score para cada dígito
print("Informe de clasificación")
print(classification_report(y_test, y_pred))
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:58:07 2025

@author: David Córdova
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ---- 1. Cargar dataset ----
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# ---- 2. Normalizar datos (0–255 -> 0–1) ----
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# ---- 3. Separar entrenamiento / validación ----
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Expandir dimensión para añadir canal (1 para escala de grises)
X_train = X_train[..., tf.newaxis]
X_val = X_val[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

print(X_train.shape)  # (48000, 28, 28, 1)


# ---- 4. Etiquetas de Fashion MNIST ----
class_names = [
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]

# ---- 5. Mostrar Prendas ----
plt.figure(figsize=(10,10))
for i in range (9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i]])
plt.show()

# ---- 6. Crear la base convolucional
cnn_model = models.Sequential()
cnn_model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)))
cnn_model.add(layers.MaxPooling2D((2,2)))
cnn_model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
cnn_model.add(layers.MaxPooling2D((2,2)))
cnn_model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

# ----7. Añadir las capas densas ----
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(64, activation='relu'))
cnn_model.add(layers.Dense(10))
cnn_model.summary()


# ----8. Compilar y entrenar el modelo
cnn_model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
history = cnn_model.fit(X_train, y_train, epochs=10,
                        validation_data = (X_test,y_test))

# ----9. Gráficas de Aprendizaje
plt.plot(history.history['accuracy'], label='Precisión')
plt.plot(history.history['val_accuracy'], label = r'Precisión_{Validación}')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.ylim([0.8, 1])
plt.legend(loc='lower right')

test_loss, test_acc = cnn_model.evaluate(X_test,  y_test, verbose=2)

# ---- 10. Predicciones y Matriz de confusión ----
# Obtener predicciones en el set de prueba
y_pred_probs = cnn_model.predict(X_test)          # probabilidades (logits)
y_pred = np.argmax(y_pred_probs, axis=1)          # clase con mayor probabilidad

# Matriz de confusión
cm_cnn = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
plt.title("Matriz de Confusión - CNN")
plt.show()

# ---- 11. Reporte de Clasificación ----
print(classification_report(y_test, y_pred, target_names=class_names))
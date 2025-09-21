# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:36:39 2025

@author: super
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ---- 1. Cargar dataset ----
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# ---- 2. Normalizar datos (0–255 -> 0–1) ----
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# ---- 3. Aplanar imágenes 28x28 → 784 ----
X_train_full = X_train_full.reshape(len(X_train_full), -1)
X_test = X_test.reshape(len(X_test), -1)

# ---- 4. Separar entrenamiento / validación ----
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ---- 5. Etiquetas de Fashion MNIST ----
class_names = [
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]

# ---- 6. Entenamiento de MLP ----
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_val)

# ---- 7. Display de Matriz de confusión ----
cm_mlp = confusion_matrix(y_val, y_pred_mlp)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp.plot(cmap = 'Blues', values_format='d')
plt.title('Matriz de Confusión de MLP')
plt.xlabel('Valor Predecido')
plt.ylabel('Valor Verdadero')
plt.show()

# ---- 8. Reporte de Clasificación ----
print(classification_report(y_val, y_pred_mlp, target_names=[
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]))
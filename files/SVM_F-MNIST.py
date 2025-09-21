# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:12:39 2025

@author: David Córdova
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ---- 1. Cargar dataset ----
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# ---- 2. Normalizar datos (0–255 -> 0–1) ----
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# ---- 3. Aplanar imágenes 28x28 → 784 ----
X_train_full_svm = X_train_full.reshape(len(X_train_full), -1)
X_test_svm = X_test.reshape(len(X_test), -1)

# ---- 4. Separar entrenamiento / validación ----
X_train_svm, X_val_svm, y_train, y_val = train_test_split(
    X_train_full_svm, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ---- 5. Etiquetas de Fashion MNIST ----
class_names = [
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]

# ---- 6. Entrenamiento de SVM
SVM = SVC(kernel = 'linear',random_state=42)
SVM.fit(X_train_svm,y_train)
y_pred_svm = SVM.predict(X_val_svm)

# ---- 7. Display de Matriz de confusión ----
cm_svm = confusion_matrix(y_val, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp.plot(cmap = 'Blues', values_format='d')
plt.title('Matriz de Confusión de SVM')
plt.xlabel('Valor Predecido')
plt.ylabel('Valor Verdadero')
plt.show()

# ---- 8. Reporte de Clasificación ----
print(classification_report(y_val, y_pred_svm, target_names=[
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]))
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 06:51:02 2025

@author: David Córdova
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# ---- 1. Cargar dataset ----
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# ---- 2. Normalizar datos (0–255 -> 0–1) ----
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# ---- 3. Aplanar imágenes 28x28 → 784 ----
X_train_full_log_reg = X_train_full.reshape(len(X_train_full), -1)
X_test_log_reg = X_test.reshape(len(X_test), -1)

# ---- 4. Separar entrenamiento / validación ----
X_train_log_reg, X_val_log_reg, y_train, y_val = train_test_split(
    X_train_full_log_reg, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ---- 5. Etiquetas de Fashion MNIST ----
class_names = [
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]

# ---- 6. Mostrar una figura con rótulo ----
idx = np.random.randint(0, len(X_train_full))  # índice aleatorio
plt.imshow(X_train_full[idx], cmap="gray")
plt.title(f"Etiqueta: {class_names[y_train_full[idx]]}")
plt.axis("off")
plt.show()

# ---- 7. Entrenamiento y Evaluación de Regresión Logística ----
log_reg = LogisticRegression(
    solver = 'saga', n_jobs=1)
log_reg.fit(X_train_log_reg, y_train)

y_pred_log_reg =  log_reg.predict(X_val_log_reg)

# ---- 8. Display de Matriz de confusión ----
cm_log_reg = confusion_matrix(y_val, y_pred_log_reg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_log_reg)
disp.plot(cmap = 'Blues', values_format='d')
plt.title('Matriz de Confusión de Regresión Logistica')
plt.xlabel('Valor Predecido')
plt.ylabel('Valor Verdadero')
plt.show()

# ---- 9. Reporte de Clasificación ----
print(classification_report(y_val, y_pred_log_reg, target_names=[
    "Camiseta", "Pantalon", "Suéter", "Vestido", "Abrigo",
    "Sandalía", "Camisa", "Zapatillas", "Bolso", "Botas"
]))
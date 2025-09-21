Proyecto 1: Clasificación con Fashion-MNIST 🧥👟

Este proyecto explora y compara el rendimiento de varios algoritmos de Machine Learning para clasificar imágenes del dataset Fashion-MNIST. Se implementan modelos que van desde algoritmos clásicos hasta redes neuronales profundas.

🎯 Objetivo

El objetivo principal es entrenar y evaluar diferentes modelos de clasificación para determinar cuál ofrece el mejor rendimiento en la tarea de identificar 10 tipos distintos de prendas de vestir a partir de imágenes en escala de grises.

Dataset

Se utiliza el dataset Fashion-MNIST, que consta de:

    70,000 imágenes de 28x28 píxeles en escala de grises.

    60,000 imágenes para el conjunto de entrenamiento.

    10,000 imágenes para el conjunto de prueba.

    10 clases de artículos de moda (camisetas, pantalones, vestidos, etc.).

🧠 Modelos Implementados

Se evaluaron los siguientes modelos:

    Regresión Logística: Un modelo lineal simple como línea base.

    Support Vector Machine (SVM): Un clasificador de vectores de soporte para encontrar el hiperplano óptimo.

    Random Forest: Un modelo de ensamble basado en árboles de decisión.

    Perceptrón Multicapa (MLP): Una red neuronal artificial de tipo feedforward.

    Red Neuronal Convolucional (CNN): Una arquitectura de red neuronal profunda especializada en el procesamiento de imágenes.

📁 Estructura del Proyecto

    .
    ├── P1_F-MNIST.ipynb        # Notebook principal con el análisis y comparación de modelos.
    └── files/                  # Directorio con las implementaciones de cada modelo.
        ├── CNN_F-MNIST.py
        ├── Log_Reg_Proyecto_1.py
        ├── MLP_F-MNIST.py
        ├── RndmFrst.ipynb
        └── SVM_F-MNIST.py

P1_F-MNIST.ipynb: Es el notebook principal. Contiene la carga de datos, el preprocesamiento, la ejecución de los modelos y la visualización de los resultados comparativos.

files/: Contiene los scripts y notebooks individuales para la implementación de cada modelo.

🛠️ Requisitos

Para ejecutar este proyecto, necesitarás tener instaladas las siguientes librerías de Python:

    TensorFlow / Keras

    Scikit-learn

    NumPy

    Pandas

    Matplotlib

    Jupyter Notebook


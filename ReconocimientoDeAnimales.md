# Clasificador de Animales con Redes Neuronales Convolucionales (CNN)

## Descripción del Proyecto

Este proyecto implementa un sistema de **clasificación de imágenes de animales** utilizando una **Red Neuronal Convolucional (CNN)** desarrollada con **TensorFlow y Keras**. El modelo es capaz de identificar imágenes pertenecientes a cinco clases distintas: gato, hormiga, mariquita, perro y tortuga.

El proyecto cubre todo el flujo de trabajo típico en visión por computadora:
- Carga y preparación del dataset.
- Aumento de datos (data augmentation).
- Construcción y entrenamiento del modelo CNN.
- Evaluación con validación.
- Guardado del modelo entrenado.
- Uso del modelo para realizar predicciones sobre imágenes externas.

El objetivo principal es **académico y didáctico**, orientado a comprender cómo funcionan las CNN aplicadas a clasificación de imágenes.

---

## Clases del Modelo

El modelo fue entrenado para clasificar las siguientes categorías:

- gato
- hormiga
- mariquita
- perro
- tortuga

Cada imagen de entrada es asignada a una de estas cinco clases.

---

## Tecnologías Utilizadas

- Python 3
- TensorFlow
- Keras
- NumPy
- Redes Neuronales Convolucionales (CNN)

---

## Preparación del Dataset

El dataset se organiza en carpetas, donde cada carpeta corresponde a una clase:

```
dataset_animales/
├── gato/
├── hormiga/
├── mariquita/
├── perro/
└── tortuga/
```

El conjunto de datos se divide automáticamente en:
- 80% entrenamiento
- 20% validación

Las imágenes se redimensionan a **32x32 píxeles** y se procesan en lotes (batch) de 32 imágenes.

---

## Arquitectura del Modelo

La red neuronal convolucional está compuesta por:

- Capa de entrada (32x32x3).
- Aumento de datos (rotación, zoom y volteo horizontal).
- Normalización de valores de píxeles.
- Tres capas convolucionales:
  - Conv2D (32 filtros)
  - Conv2D (64 filtros)
  - Conv2D (128 filtros)
- Capas MaxPooling para reducción espacial.
- Capa Flatten.
- Capa Dense de 64 neuronas con activación ReLU.
- Dropout del 50% para evitar sobreajuste.
- Capa de salida con 5 neuronas y activación Softmax.

La función de pérdida utilizada es **sparse_categorical_crossentropy** y el optimizador es **Adam**.

---

## Entrenamiento del Modelo

El modelo se entrena durante un máximo de 30 épocas, utilizando **Early Stopping** para detener el entrenamiento si la pérdida de validación no mejora durante 3 épocas consecutivas.

Al finalizar el entrenamiento, el modelo se guarda en un archivo `.h5` para su uso posterior.

---

## Predicción de Imágenes

El sistema incluye un script interactivo que permite:
- Cargar el modelo entrenado.
- Ingresar la ruta de una imagen.
- Obtener el porcentaje de probabilidad para cada clase.
- Mostrar la clase predicha como resultado final.

El programa se ejecuta en consola y permite realizar múltiples predicciones hasta que el usuario escriba `salir`.

---

## Estructura del Proyecto

```
.
├── entrenamiento.py
├── prediccion.py
├── modelo_animales_cnn.h5
└── README.md
```

---

## Propósito Académico

Este proyecto fue desarrollado con fines académicos para:
- Comprender el funcionamiento de las redes neuronales convolucionales.
- Aplicar técnicas de visión por computadora.
- Analizar procesos de entrenamiento, validación y predicción.
- Reforzar conceptos de Inteligencia Artificial y Deep Learning.

---

## Código Fuente

### Entrenamiento del Modelo CNN

```python
import tensorflow as tf
from tensorflow.keras import layers, models

ruta_datos = r'C:\Users\wiki2\Documents\ProyectoAnimales\dataset_animales'
IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 30

clases = ['gato', 'hormiga', 'mariquita', 'perro', 'tortuga']

print("Cargando datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_datos,
    class_names=clases,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    ruta_datos,
    class_names=clases,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=123
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

model.save('modelo_animales_cnn.h5')
print("Modelo guardado correctamente")
```

---

### Predicción con el Modelo Entrenado

```python
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model('modelo_animales_cnn.h5')
clases = ['gato', 'hormiga', 'mariquita', 'perro', 'tortuga']
IMG_SIZE = 32

def predecir(ruta):
    ruta_limpia = ruta.strip().replace("& ", "").replace("'", "").replace('"', "")
    if not os.path.exists(ruta_limpia):
        return

    img = tf.keras.utils.load_img(ruta_limpia, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    ganador = np.argmax(preds)

    print("\n" + "="*30)
    for i in range(5):
        print(f"{clases[i]}: {preds[i]*100:.1f}%")
    print(f"\nIA DICE: {clases[ganador].upper()}")
    print("="*30)

while True:
    r = input("\nImagen: ")
    if r.lower() == 'salir':
        break
    predecir(r)
```

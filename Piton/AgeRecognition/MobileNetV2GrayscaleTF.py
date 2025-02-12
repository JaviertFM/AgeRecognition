#Tensorflow 10%

import os
import random
import numpy as np
from skimage import io, color, exposure
from skimage.transform import rotate
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.config.list_physical_devices())

# Parámetros
IMG_SIZE = (224, 224)  # Tamaño de entrada de la red
SUBSET_PERCENTAGE = 1  # Usar el 10% del dataset


# Preprocesado básico
def preprocess_image(image_path):
    image = io.imread(image_path)
    gray = color.rgb2gray(image)  # Convertir a escala de grises
    eq_img = exposure.equalize_hist(gray)  # Ecualización de histograma
    denoised = cv2.fastNlMeansDenoising((eq_img * 255).astype(np.uint8), None, 10, 7, 21)
    resized = cv2.resize(denoised, IMG_SIZE)
    return resized


# Función para cargar un subconjunto del dataset
def load_subset_dataset(base_dir, subset_percentage):
    images = []
    labels = []
    all_image_paths = []

    # Recorremos las carpetas de rangos de edad
    for age_range in os.listdir(base_dir):
        range_dir = os.path.join(base_dir, age_range, 'original')
        if not os.path.isdir(range_dir):
            continue  # Evita archivos no deseados
        label = int(age_range.split('_')[1])  # Edad mínima como etiqueta

        # Recopilar todas las imágenes en la carpeta original
        for img_file in os.listdir(range_dir):
            img_path = os.path.join(range_dir, img_file)
            all_image_paths.append((img_path, label))

    # Seleccionar aleatoriamente el 10% del dataset
    random.shuffle(all_image_paths)
    subset_size = int(len(all_image_paths) * subset_percentage)
    subset = all_image_paths[:subset_size]

    # Preprocesar imágenes seleccionadas
    for img_path, label in subset:
        preprocessed = preprocess_image(img_path)
        images.append(preprocessed)
        labels.append(label)

    return np.array(images), np.array(labels)


# Directorio del dataset
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'

# Cargar solo el 10% del dataset
images, labels = load_subset_dataset(base_dir, SUBSET_PERCENTAGE)

# Normalizar imágenes
images = images / 255.0

# Imprimir información
print(f"Número total de imágenes cargadas: {len(images)}")
print(f"Dimensiones de las imágenes: {images.shape}")
print(f"Ejemplo de etiquetas: {np.unique(labels)}")




# Configuración de parámetros
IMG_SIZE = (224, 224)  # Tamaño de imagen
EPOCHS = 10
BATCH_SIZE = 16

# Callback personalizado para mostrar progreso
class CustomProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🕒 Epoch {epoch + 1}/{EPOCHS} comenzando...")

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        accuracy = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        print(f"✅ Epoch {epoch + 1} terminada.")
        print(f"   🔹 Pérdida de entrenamiento: {loss:.4f} | Precisión de entrenamiento: {accuracy:.2%}")
        print(f"   🔹 Pérdida de validación: {val_loss:.4f} | Precisión de validación: {val_accuracy:.2%}")

# Cargar imágenes y etiquetas (simulado aquí como imágenes aleatorias)
num_classes = 5  # Ejemplo con 5 clases de edades
num_samples = 100  # Simulamos un dataset pequeño
X = np.random.rand(num_samples, *IMG_SIZE, 1)  # Imágenes aleatorias
y = np.random.randint(0, num_classes, size=num_samples)  # Etiquetas aleatorias

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Crear el modelo base
base_model = MobileNetV2(weights=None, include_top=False, input_tensor=Input(shape=(*IMG_SIZE, 1)))

# Capas superiores personalizadas
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Compilación del modelo
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento con barra de progreso y callback
print("🚀 Iniciando el entrenamiento del modelo...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[CustomProgressCallback()]
)


# Visualización de precisión y pérdida
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

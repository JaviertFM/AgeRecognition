# overflow de memoria can't allocate 176gb

import os
import random
import numpy as np
from skimage import io, color, exposure
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf

# Parámetros
IMG_SIZE = (224, 224)  # Tamaño de entrada de la red
SUBSET_PERCENTAGE = 1  # Usar el 100% del dataset
BATCH_SIZE = 32
EPOCHS = 30
NUM_PARTS = 11  # Número de partes (barbilla, boca, cejas, etc.)
INPUT_SHAPE = (*IMG_SIZE, NUM_PARTS + 1)  # +1 para la imagen original

# Preprocesado básico
def preprocess_image(image_path):
    image = io.imread(image_path)
    gray = color.rgb2gray(image)  # Convertir a escala de grises
    eq_img = exposure.equalize_hist(gray)  # Ecualización de histograma
    denoised = cv2.fastNlMeansDenoising((eq_img * 255).astype(np.uint8), None, 10, 7, 21)
    resized = cv2.resize(denoised, IMG_SIZE)
    return resized

# Cargar imágenes originales y partes asociadas
def load_dataset_with_parts(base_dir, subset_percentage):
    images, labels = [], []
    all_image_data = []

    # Directorios de partes asociadas
    parts_dirs = ['barbilla', 'boca', 'ceja_derecho', 'ceja_izquierdo',
                  'ceno', 'frente', 'mejilla_derecho', 'mejilla_izquierdo',
                  'nariz', 'ojo_derecho', 'ojo_izquierdo']

    for age_range in os.listdir(base_dir):
        range_dir = os.path.join(base_dir, age_range)
        if not os.path.isdir(range_dir):
            continue
        label = int(age_range.split('_')[1])  # Edad mínima como etiqueta

        # Cargar imagen original y partes asociadas
        original_dir = os.path.join(range_dir, 'original')
        for img_file in os.listdir(original_dir):
            img_name, ext = os.path.splitext(img_file)
            img_path = os.path.join(original_dir, img_file)

            # Imagen original
            image_set = [preprocess_image(img_path)]

            # Añadir partes asociadas
            for part in parts_dirs:
                part_path = os.path.join(range_dir, part, f"{img_name}_{part}.jpg")
                if os.path.exists(part_path):
                    image_set.append(preprocess_image(part_path))
                else:
                    image_set.append(np.zeros(IMG_SIZE))  # Si falta, usar imagen negra

            # Combinar las imágenes en un solo array (canales adicionales)
            combined_image = np.stack(image_set, axis=-1)  # (224, 224, NUM_PARTS + 1)
            all_image_data.append((combined_image, label))

    # Seleccionar aleatoriamente un subset del dataset
    random.shuffle(all_image_data)
    subset_size = int(len(all_image_data) * subset_percentage)
    subset = all_image_data[:subset_size]

    # Separar imágenes y etiquetas
    for image, label in subset:
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

# Cargar el dataset
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'
images, labels = load_dataset_with_parts(base_dir, SUBSET_PERCENTAGE)

# Normalizar imágenes
images = images / 255.0
print(f"Número total de imágenes: {len(images)}, Dimensión de entrada: {images.shape}, Clases: {np.unique(labels)}")

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# Callbacks personalizados
class CustomProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        accuracy = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_accuracy = logs.get("val_accuracy")
        print(f"✅ Epoch {epoch + 1} completada | "
              f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2%}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Crear el modelo
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
for layer in base_model.layers:
    layer.trainable = False  # Congelar el modelo base inicialmente

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(np.unique(labels)), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    callbacks=[CustomProgressCallback(), early_stopping, model_checkpoint])

# Graficar resultados
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

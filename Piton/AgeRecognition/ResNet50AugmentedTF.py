#Tensorflow todo overflow

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from skimage import io, color, exposure
from skimage.transform import rotate
import cv2


# 1. Parámetros
IMG_SIZE = (224, 224)  # Tamaño de entrada de la red
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# 2. Preprocesado de imágenes (función personalizada)
def preprocess_image(image_path):
    image = io.imread(image_path)
    # Convertir a escala de grises (opcional: R, G, B o YUV)
    gray = color.rgb2gray(image)
    # Ecualización del histograma para mejorar contraste
    eq_img = exposure.equalize_hist(gray)
    # Suavizado (denoise)
    denoised = cv2.fastNlMeansDenoising((eq_img * 255).astype(np.uint8), None, 10, 7, 21)
    # Resize a tamaño estándar
    resized = cv2.resize(denoised, IMG_SIZE)
    return resized

# 3. Augmentación personalizada
def augment_image(image):
    augmented = []
    # Rotación
    augmented.append(rotate(image, angle=random.uniform(-15, 15), mode='wrap'))
    # Flip horizontal
    augmented.append(np.fliplr(image))
    # Brillo
    augmented.append(cv2.convertScaleAbs(image, alpha=1.0, beta=random.uniform(-30, 30)))
    return augmented

# 4. Cargar datos del dataset
def load_dataset(base_dir):
    images = []
    labels = []
    for age_range in os.listdir(base_dir):  # Ej. "age_0_4"
        range_dir = os.path.join(base_dir, age_range, 'original')
        label = int(age_range.split('_')[1])  # Edad mínima como etiqueta
        for img_file in os.listdir(range_dir):
            img_path = os.path.join(range_dir, img_file)
            preprocessed = preprocess_image(img_path)
            images.append(preprocessed)
            labels.append(label)
            # Aplicar augmentación
            for aug_img in augment_image(preprocessed):
                images.append(aug_img)
                labels.append(label)
    return np.array(images), np.array(labels)

# 5. Crear el modelo
def create_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# 6. Cargar datos
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'
images, labels = load_dataset(base_dir)

# Normalizar imágenes
images = images / 255.0

# Dividir en conjuntos de entrenamiento y validación
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# Calcular pesos de clase para balancear
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# 7. Entrenar modelo
model = create_model(input_shape=(*IMG_SIZE, 1), num_classes=len(np.unique(labels)))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenamiento
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          class_weight=class_weights,
          callbacks=[early_stopping])

# Guardar modelo
model.save('age_estimation_model.h5')

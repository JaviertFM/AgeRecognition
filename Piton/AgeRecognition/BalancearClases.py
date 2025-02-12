import os
import shutil
import random
from pathlib import Path
from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ====== PARÁMETROS ======
SOURCE_DIR = Path('dataset_rangos_10')  # Dataset reorganizado
BALANCED_DIR = Path('dataset_rangos_10_balanced')  # Carpeta destino balanceada
TARGET_SIZE = 5000  # Tamaño objetivo por clase
IMG_SIZE = (224, 224)  # Tamaño estándar de imágenes
SEED = 42  # Para reproducibilidad

# ====== CREAR DIRECTORIOS ======
BALANCED_DIR.mkdir(parents=True, exist_ok=True)

# ====== FUNCIONES ======

def count_images(folder):
    """Contar imágenes en una carpeta."""
    return len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])


def augment_image(image_path, save_path, augmentation_gen, n_augment):
    """Generar imágenes aumentadas para balancear las clases."""
    # Cargar imagen
    img = Image.open(image_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)

    # Expandir dimensiones para generar aumentaciones
    img_array = np.expand_dims(img_array, axis=0)

    # Generar augmentaciones
    count = 0
    for batch in augmentation_gen.flow(img_array, batch_size=1, save_to_dir=save_path, save_prefix="aug", save_format="jpg"):
        count += 1
        if count >= n_augment:
            break


def balance_class(class_path, save_path, target_size, augmentation_gen):
    """Balancear una clase dada, generando imágenes aumentadas si es necesario."""
    # Contar imágenes actuales
    current_count = count_images(class_path)

    # Si ya tiene suficientes imágenes, solo copiar
    if current_count >= target_size:
        print(f"✅ Clase ya balanceada: {class_path}")
        for img_file in os.listdir(class_path):
            shutil.copy2(os.path.join(class_path, img_file), save_path)
        return

    # Copiar imágenes existentes
    print(f"⚠️ Clase desbalanceada: {class_path}. Aumentando...")
    images = os.listdir(class_path)
    for img_file in images:
        shutil.copy2(os.path.join(class_path, img_file), save_path)

    # Generar augmentaciones para alcanzar el target_size
    n_augment = (target_size - current_count) // len(images)
    for img_file in images:
        img_path = os.path.join(class_path, img_file)
        augment_image(img_path, save_path, augmentation_gen, n_augment)


# ====== DATA AUGMENTATION CONFIG ======
augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# ====== PROCESAR CLASES ======
for folder in os.listdir(SOURCE_DIR):
    class_path = SOURCE_DIR / folder
    save_path = BALANCED_DIR / folder
    save_path.mkdir(parents=True, exist_ok=True)

    # Balancear cada clase
    balance_class(class_path, save_path, TARGET_SIZE, augmentation)

print("📊 Dataset balanceado correctamente.")

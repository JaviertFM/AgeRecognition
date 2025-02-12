import os
import random
import numpy as np
from skimage import io, color, exposure
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Parámetros
IMG_SIZE = (224, 224)  # Tamaño de imagen
BATCH_SIZE = 32  # Tamaño pequeño debido a 16 GB de RAM
EPOCHS = 80
INPUT_SHAPE = (*IMG_SIZE, 3)  # 3 canales para RGB

# Preprocesado básico
def preprocess_image(image_path):
    """Preprocesar una imagen: redimensionar y convertir a RGB."""
    try:
        image = io.imread(image_path)  # Cargar la imagen
        if len(image.shape) == 2:  # Si está en escala de grises
            image = color.gray2rgb(image)  # Convertir a RGB
        resized = cv2.resize(image, IMG_SIZE)  # Redimensionar
        return resized
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)  # Imagen vacía en caso de error

# Generador de datos
class DataGenerator(Sequence):
    """Generador personalizado para cargar imágenes originales en tiempo real."""
    def __init__(self, base_dir, batch_size, img_size, shuffle=True):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.image_paths = self._get_image_paths()
        self.on_epoch_end()

    def _get_image_paths(self):
        image_paths = []
        for age_range in os.listdir(self.base_dir):
            range_dir = os.path.join(self.base_dir, age_range)
            label = int(age_range.split('_')[1])  # Edad mínima como etiqueta
            original_dir = os.path.join(range_dir, 'original')
            for img_file in os.listdir(original_dir):
                img_path = os.path.join(original_dir, img_file)
                image_paths.append((img_path, label))
        return image_paths

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = [], []
        for img_path, label in batch_paths:
            images.append(preprocess_image(img_path))
            labels.append(label)
        return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int32)  # Normalizar imágenes

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_paths)

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

# Directorio base del dataset
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'

# Crear generadores de datos
train_generator = DataGenerator(base_dir, BATCH_SIZE, IMG_SIZE)
val_generator = DataGenerator(base_dir, BATCH_SIZE, IMG_SIZE, shuffle=False)

# Crear el modelo con MobileNetV2 y pesos preentrenados
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=INPUT_SHAPE))

# Congelar capas base inicialmente
for layer in base_model.layers:
    layer.trainable = False

# Capas superiores personalizadas
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(set(train_generator.image_paths)), activation='softmax')(x)  # Número de clases basado en etiquetas

# Crear el modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001),  # Learning rate pequeño
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model_v2.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Entrenar el modelo
print("🚀 Iniciando el entrenamiento del modelo...\n")
history = model.fit(train_generator,
                    validation_data=val_generator,
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

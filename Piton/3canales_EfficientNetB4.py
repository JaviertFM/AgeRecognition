import os
import numpy as np
import random
from sklearn.model_selection import train_test_split  # Importado correctamente
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
from skimage import io, color
import cv2
import matplotlib.pyplot as plt

# TensorFlow y Keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros
IMG_SIZE = (224, 224)  # Tamaño de las imágenes
BATCH_SIZE = 16        # Reducido para evitar OOM
EPOCHS = 80            # Número de épocas
INPUT_SHAPE = (*IMG_SIZE, 3)
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# **Paso 1: Crear el mapa de clases agrupadas**
def create_label_map(base_dir):
    """Agrupar carpetas en clases de 10 años (0-9, 10-19, etc.)."""
    class_names = sorted(os.listdir(base_dir))  # Carpetas originales
    label_map = {}
    for class_name in class_names:
        age_range = class_name.split('_')[1]  # Extraer rango (e.g., "0_5")
        start_age = int(age_range.split('_')[0])
        group = (start_age // 10) * 10  # Agrupar en décadas
        group_name = f"age_{group}_{group+9}"
        if group_name not in label_map:
            label_map[group_name] = len(label_map)
    return label_map

# Preprocesado básico
def preprocess_image(image_path):
    """Redimensionar y convertir a RGB."""
    try:
        image = io.imread(image_path)
        if len(image.shape) == 2:  # Imagen en escala de grises
            image = color.gray2rgb(image)
        resized = cv2.resize(image, IMG_SIZE)  # Redimensionar
        return resized
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)

# **Paso 2: Generador de datos dinámico**
from tensorflow.keras.utils import Sequence  # Importar Sequence

class DataGenerator(Sequence):  # Heredar de Sequence
    """Generador dinámico para cargar imágenes por lotes con augmentación."""
    def __init__(self, image_paths, labels, batch_size, augmentations=None, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Número total de lotes por época."""
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        """Generar un lote de datos."""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for path in batch_paths:
            image = preprocess_image(path)  # Procesar imagen
            if self.augmentations:  # Aplicar augmentación si está activada
                image = self.augmentations.random_transform(image)
            images.append(image)

        # Normalizar las imágenes al rango [0, 1]
        images = np.array(images) / 255.0
        return images, np.array(batch_labels)

    def on_epoch_end(self):
        """Barajar los datos después de cada época."""
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
# Callbacks personalizados
class CustomProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"✅ Epoch {epoch + 1} completada | Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.2%}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Acc: {logs['val_accuracy']:.2%}")

# **Paso 3: Cargar y dividir datos**
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'
label_map = create_label_map(base_dir)
print("Mapa de etiquetas:", label_map)

# Preparar rutas e imágenes
image_paths, labels = [], []
for folder in os.listdir(base_dir):
    folder_range = folder.split('_')[1:3]  # Obtener el rango de la carpeta
    folder_start = int(folder_range[0])
    folder_end = int(folder_range[1])

    # Buscar el grupo correcto en el mapa
    for group_name, label in label_map.items():
        group_start = int(group_name.split('_')[1])
        group_end = int(group_name.split('_')[2])
        if folder_start >= group_start and folder_end <= group_end:
            folder_path = os.path.join(base_dir, folder, 'original')
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    image_paths.append(os.path.join(folder_path, img_file))
                    labels.append(label)

# Dividir en conjuntos
X_train_val, X_test, y_train_val, y_test = train_test_split(
    image_paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SPLIT, stratify=y_train_val, random_state=42
)

# Augmentación
augmentations = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Generadores dinámicos
train_generator = DataGenerator(X_train, y_train, BATCH_SIZE, augmentations)
val_generator = DataGenerator(X_val, y_val, BATCH_SIZE, augmentations, shuffle=False)
test_generator = DataGenerator(X_test, y_test, BATCH_SIZE, None, shuffle=False)

# **Paso 4: Modelo EfficientNetB4**
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
for layer in base_model.layers[-100:]:
    layer.trainable = True

x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(len(label_map), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# **Paso 5: Entrenamiento**
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[
    CustomProgressCallback(),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_model_effnetb4_gen.h5', monitor='val_loss', save_best_only=True, verbose=1)
])

# **Evaluación**
y_pred = np.argmax(model.predict(test_generator), axis=1)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

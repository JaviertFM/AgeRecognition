import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from skimage import io, color, exposure
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 80
INPUT_SHAPE = (*IMG_SIZE, 3)
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# Crear un mapa de clases
def create_label_map(base_dir):
    """Crea un mapa de clases basado en las carpetas."""
    class_names = sorted(os.listdir(base_dir))
    label_map = {class_name: idx for idx, class_name in enumerate(class_names)}
    return label_map

# Preprocesado básico
def preprocess_image(image_path):
    """Preprocesar una imagen: redimensionar y convertir a RGB."""
    try:
        image = io.imread(image_path)
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        resized = cv2.resize(image, IMG_SIZE)
        return resized
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)

# Cargar y dividir los datos
def load_and_split_data(base_dir, label_map, test_split, val_split):
    """Cargar los datos y dividirlos en entrenamiento, validación y prueba."""
    image_paths, labels = [], []

    for class_name, label in label_map.items():
        class_dir = os.path.join(base_dir, class_name, 'original')
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            image_paths.append(img_path)
            labels.append(label)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split, stratify=y_train_val, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Generador de datos con augmentación condicional para clases minoritarias
def augment_minority_classes(X_train, y_train, target_size, augmentation):
    """Aplicar augmentación a clases minoritarias para balancear el dataset."""
    class_counts = Counter(y_train)
    minority_classes = [cls for cls, count in class_counts.items() if count < target_size]
    augmented_images = []
    augmented_labels = []

    # Preprocesar imágenes originales
    X_train_preprocessed = np.array([preprocess_image(img_path) for img_path in X_train])

    for class_label in minority_classes:
        # Filtrar imágenes de la clase minoritaria
        class_images = [X_train_preprocessed[i] for i in range(len(y_train)) if y_train[i] == class_label]

        # Generar imágenes augmentadas hasta alcanzar el tamaño objetivo
        while len(class_images) + len(augmented_images) < target_size:
            for img in class_images:
                img = np.expand_dims(img, axis=0)  # Expandir dimensiones para el generador
                aug_iter = augmentation.flow(img, batch_size=1)
                aug_img = next(aug_iter)[0].astype(np.float32)
                augmented_images.append(aug_img)
                augmented_labels.append(class_label)
                if len(augmented_images) >= (target_size - len(class_images)):
                    break

    # Concatenar datos originales y augmentados
    X_train_balanced = np.concatenate([X_train_preprocessed, np.array(augmented_images)], axis=0)
    y_train_balanced = np.concatenate([y_train, np.array(augmented_labels)], axis=0)

    return shuffle(X_train_balanced, y_train_balanced, random_state=42)



class DataGenerator(Sequence):
    """Generador personalizado para cargar imágenes en lotes con soporte de augmentación."""

    def __init__(self, image_paths, labels, batch_size, img_size, augmentation=None, shuffle=True):
        self.image_paths = image_paths  # Lista de rutas de imágenes o imágenes ya procesadas
        self.labels = labels  # Etiquetas correspondientes
        self.batch_size = batch_size
        self.img_size = img_size
        self.augmentation = augmentation  # Generador de augmentación de datos (opcional)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Número total de lotes en cada época."""
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        """Generar un lote de datos."""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        images, labels = [], []

        for img, label in zip(batch_paths, batch_labels):
            # Si las imágenes están como rutas, cárgalas y preprocésalas
            if isinstance(img, str):
                img = preprocess_image(img)

            # Aplicar augmentación si está definida
            if self.augmentation:
                img = self.augmentation.random_transform(img)

            images.append(img)
            labels.append(label)

        return np.array(images, dtype=np.float32) / 255.0, np.array(labels, dtype=np.int32)

    def on_epoch_end(self):
        """Barajar los datos después de cada época (opcional)."""
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)


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

class EarlyStoppingByLR(Callback):
    def __init__(self, min_lr=1e-6):
        super(EarlyStoppingByLR, self).__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr.numpy()
        if lr <= self.min_lr:
            print(f"⚠️ Deteniendo el entrenamiento: el learning rate alcanzó el mínimo ({self.min_lr})")
            self.model.stop_training = True

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

# Directorio base del dataset
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'

# Crear el mapa de clases
label_map = create_label_map(base_dir)
print("Mapa de etiquetas:", label_map)

# Cargar y dividir los datos
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(base_dir, label_map, TEST_SPLIT, VAL_SPLIT)

# Crear aumentación de datos
augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Balancear dataset con augmentación condicional
target_size = max(Counter(y_train).values())  # Igualar al tamaño de la clase mayoritaria
X_train_balanced, y_train_balanced = augment_minority_classes(X_train, y_train, target_size, augmentation)

# Crear generadores de datos
train_generator = DataGenerator(X_train_balanced, y_train_balanced, BATCH_SIZE, IMG_SIZE, augmentation=augmentation)
val_generator = DataGenerator(X_val, y_val, BATCH_SIZE, IMG_SIZE)
test_generator = DataGenerator(X_test, y_test, BATCH_SIZE, IMG_SIZE, shuffle=False)

# Crear el modelo con ResNet50 preentrenada
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=INPUT_SHAPE))
for layer in base_model.layers[-100:]:  # Descongelar últimas 100 capas
    layer.trainable = True

# Capas superiores personalizadas
# Capas superiores personalizadas con mayor capacidad
x = Flatten()(base_model.output)
x = Dense(2048, activation='relu')(x)  # Más neuronas en la primera capa densa
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)  # Segunda capa con mayor capacidad
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)  # Una capa adicional
x = Dropout(0.2)(x)
output = Dense(len(label_map), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


# Compilar el modelo con un learning rate más alto
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model_resnet_finetune.h5', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
early_stopping_lr = EarlyStoppingByLR(min_lr=1e-6)

callbacks = [CustomProgressCallback(), early_stopping, model_checkpoint, reduce_lr, lr_scheduler, early_stopping_lr]

# Entrenar el modelo
print("🚀 Iniciando el entrenamiento del modelo...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluar en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}")

# Graficar resultados
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('Precisión por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

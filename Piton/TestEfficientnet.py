import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# TensorFlow y Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from skimage import io, color
import cv2

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INPUT_SHAPE = (*IMG_SIZE, 3)

# **Paso 1: Cargar el Modelo Guardado**
model = load_model('best_model_effnetb4_gen.h5')  # Cargar el modelo guardado
def preprocess_image(image_path):
    """Redimensionar y convertir a RGB."""
    try:
        image = io.imread(image_path)
        if len(image.shape) == 2:  # Convertir escala de grises a RGB
            image = color.gray2rgb(image)
        resized = cv2.resize(image, IMG_SIZE)  # Redimensionar
        return resized
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return np.zeros((*IMG_SIZE, 3), dtype=np.float32)
class DataGenerator(Sequence):
    """Generador para cargar imágenes por lotes."""
    def __init__(self, image_paths, labels, batch_size, augmentations=None, shuffle=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Número total de lotes."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))  # Garantizar cubrir todas las muestras

    def __getitem__(self, index):
        """Obtener un lote."""
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        for path in batch_paths:
            image = preprocess_image(path)  # Preprocesar imagen
            images.append(image)

        images = np.array(images) / 255.0  # Normalización
        return images, np.array(batch_labels)

    def on_epoch_end(self):
        """Barajar después de cada época (opcional)."""
        if self.shuffle:
            combined = list(zip(self.image_paths, self.labels))
            np.random.shuffle(combined)
            self.image_paths, self.labels = zip(*combined)
# Directorio base del dataset
base_dir = 'C:/Users/Javier/Desktop/AgeDetection/Piton/AgeRecognition/dataset_rangos'

# Cargar rutas e etiquetas
image_paths = []  # Lista para rutas de imágenes
labels = []       # Lista para etiquetas
label_map = {'age_0_9': 0, 'age_10_19': 1, 'age_20_29': 2, 'age_30_39': 3, 'age_40_49': 4,
             'age_50_59': 5, 'age_60_69': 6, 'age_70_79': 7, 'age_80_89': 8, 'age_90_99': 9,
             'age_100_109': 10, 'age_110_119': 11}  # Mismo mapa de etiquetas

# Recorrer el dataset de prueba
for folder in os.listdir(base_dir):
    folder_range = folder.split('_')[1:3]  # Obtener rango
    folder_start = int(folder_range[0])
    folder_end = int(folder_range[1])

    # Buscar el grupo correcto
    for group_name, label in label_map.items():
        group_start = int(group_name.split('_')[1])
        group_end = int(group_name.split('_')[2])
        if folder_start >= group_start and folder_end <= group_end:
            folder_path = os.path.join(base_dir, folder, 'original')
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    image_paths.append(os.path.join(folder_path, img_file))
                    labels.append(label)

# Dividir en Test
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
test_generator = DataGenerator(X_test, y_test, BATCH_SIZE, augmentations=None, shuffle=False)
# Obtener predicciones por lotes
y_pred = []
y_true = []

for i in range(len(test_generator)):
    images, labels = test_generator[i]
    preds = np.argmax(model.predict(images, verbose=0), axis=1)  # Predicción lote por lote
    y_pred.extend(preds)  # Predicciones
    y_true.extend(labels)  # Etiquetas verdaderas

# Convertir a arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Reporte de métricas
print("Classification Report:\n", classification_report(y_true, y_pred))

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='viridis')
plt.title("Matriz de Confusión")
plt.show()

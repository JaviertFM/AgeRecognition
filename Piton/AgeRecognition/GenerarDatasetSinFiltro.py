import os
import shutil
from pathlib import Path
import uuid  # Para nombres únicos

# ====== PARÁMETROS ======
SOURCE_DIR = Path('dataset_rangos')  # Dataset original
TARGET_DIR = Path('dataset_notfiltered')  # Nuevo dataset sin filtrar
GROUP_SIZE = 10  # Agrupar rangos de 10 años
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')  # Extensiones aceptadas

# ====== CREAR CARPETAS ======
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# ====== FUNCIONES ======

def get_age_range(folder_name):
    """
    Calcula el nuevo rango de 10 años basado en el nombre del rango.
    """
    try:
        age_start = int(folder_name.split('_')[1])  # Extrae el inicio del rango
        group_start = (age_start // GROUP_SIZE) * GROUP_SIZE  # Agrupa por decenas
        group_end = group_start + GROUP_SIZE - 1
        return f"{group_start}_{group_end}"
    except (IndexError, ValueError):
        return None


def process_images(src_folder, dst_folder):
    """
    Procesa las imágenes de una carpeta origen y las copia a la carpeta destino renombrándolas.
    """
    # Crear carpeta destino si no existe
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Filtrar imágenes en la raíz (no subcarpetas)
    for img_file in os.listdir(src_folder):
        img_path = src_folder / img_file

        # Procesar solo archivos de imagen válidos en la raíz
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXTENSIONS:
            # Crear nombre único usando uuid
            new_name = f"{uuid.uuid4().hex}.jpg"
            shutil.copy2(img_path, dst_folder / new_name)


# ====== PROCESAR LOS DATOS ======
for folder in os.listdir(SOURCE_DIR):
    src_folder = SOURCE_DIR / folder  # Ruta del rango original

    # Saltar si no es un directorio
    if not src_folder.is_dir():
        continue

    # Obtener el nuevo rango de 10 años
    new_range = get_age_range(folder)
    if not new_range:
        print(f"⚠️ Saltando carpeta no válida: {folder}")
        continue

    # Crear carpeta destino para el rango de 10 años
    dst_folder = TARGET_DIR / f"age_{new_range}"

    # Procesar imágenes en la raíz del rango
    process_images(src_folder, dst_folder)

print("📂 Dataset sin filtrar creado correctamente.")

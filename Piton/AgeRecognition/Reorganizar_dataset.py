import os
import shutil
from pathlib import Path
import uuid  # Para generar nombres únicos

# Rutas base
source_dir = Path('dataset_rangos')
target_dir = Path('dataset_rangos_10')

# Crear carpeta destino si no existe
target_dir.mkdir(parents=True, exist_ok=True)

# Función para calcular el nuevo rango
def get_new_range(age_start):
    new_start = (age_start // 10) * 10
    new_end = new_start + 9
    return f"{new_start}_{new_end}"

# Procesar carpetas en el origen
for folder in os.listdir(source_dir):
    folder_path = source_dir / folder

    # Verificar si es un directorio válido
    if not folder_path.is_dir():
        continue

    try:
        # Extraer rango de edad
        age_range = folder.split('_')[1:]  # ['0', '4']
        age_start = int(age_range[0])  # Edad inicial
    except (IndexError, ValueError):
        print(f"Error procesando carpeta {folder}")
        continue

    # Obtener el rango combinado (10 años)
    new_range = get_new_range(age_start)
    new_folder = target_dir / f"age_{new_range}"
    new_folder.mkdir(parents=True, exist_ok=True)

    # Copiar y renombrar imágenes
    original_path = folder_path / 'original'
    if original_path.exists():
        for img_file in os.listdir(original_path):
            src_file = original_path / img_file

            # Generar un nombre único para evitar conflictos
            new_filename = f"{uuid.uuid4().hex}.jpg"
            dest_file = new_folder / new_filename

            # Copiar la imagen
            shutil.copy2(src_file, dest_file)

print("📂 Reorganización completada sin conflictos.")

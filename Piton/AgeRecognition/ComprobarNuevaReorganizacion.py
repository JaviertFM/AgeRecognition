import os
from pathlib import Path

# Definir las rutas base
source_dir = Path('dataset_rangos')
target_dir = Path('dataset_rangos_10')

# Función para contar imágenes en una carpeta
def count_images_in_folder(folder):
    folder_path = Path(folder) / 'original'
    if folder_path.exists():
        return len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    return 0

# Función para comprobar la suma de imágenes
def check_image_count(ranges_5, range_10):
    total_images = 0

    # Sumar las imágenes de los rangos de 5 años
    for r in ranges_5:
        folder = source_dir / f"age_{r}"
        count = count_images_in_folder(folder)
        print(f"Rango {r}: {count} imágenes")
        total_images += count

    # Contar las imágenes en el rango combinado de 10 años
    combined_folder = target_dir / f"age_{range_10}"
    combined_count = len([f for f in os.listdir(combined_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Mostrar resultados
    print(f"Total esperado en {range_10}: {total_images} imágenes")
    print(f"Total encontrado en {range_10}: {combined_count} imágenes")

    # Comprobar si coinciden
    if total_images == combined_count:
        print("✅ ¡La combinación es correcta!")
    else:
        print("❌ ¡Hay un desajuste en el número de imágenes!")

# Ejemplo: Verificar el rango 0-9 (0-4 y 5-9)
check_image_count(['0_4', '5_9'], '0_9')

# Puedes repetir para otros rangos:
check_image_count(['10_14', '15_19'], '10_19')
check_image_count(['20_24', '25_29'], '20_29')
check_image_count(['30_34', '35_39'], '30_39')
check_image_count(['40_44', '45_49'], '40_49')
check_image_count(['50_54', '55_59'], '50_59')
check_image_count(['60_64', '65_69'], '60_69')
check_image_count(['70_74', '75_79'], '70_79')
check_image_count(['80_84', '85_89'], '80_89')
check_image_count(['90_94', '95_99'], '90_99')
check_image_count(['100_104', '105_109'], '100_109')
check_image_count(['110_114', '115_119'], '110_119')

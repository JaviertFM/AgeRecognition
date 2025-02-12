import os
import shutil
import re


def organizar_por_edad_FGNET(input_folder, output_base_folder="dataset"):
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # Obtener la lista de imágenes en la carpeta de entrada
    archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for archivo in archivos:
        # Extraer la edad de los dos últimos dígitos del nombre del archivo
        edad_match = re.search(r'(\d{2})(?=\.\w+$)', archivo)
        if edad_match:
            edad = edad_match.group(1)
            edad_folder = os.path.join(output_base_folder, edad)

            # Crear la carpeta de edad si no existe
            if not os.path.exists(edad_folder):
                os.makedirs(edad_folder)

            # Contador basado en la cantidad de archivos en la carpeta de edad
            contador = len(os.listdir(edad_folder)) + 1
            nuevo_nombre = f"{contador}.jpg"

            # Ruta de archivo original y nueva ruta de archivo
            origen = os.path.join(input_folder, archivo)
            destino = os.path.join(edad_folder, nuevo_nombre)

            # Copiar el archivo a la carpeta de destino
            shutil.copy2(origen, destino)
            print(f"Imagen {archivo} copiada a {destino}")
        else:
            print(f"No se pudo determinar la edad para el archivo {archivo}. Ignorando.")


def organizar_por_edad_UTKPlus(input_folder, output_base_folder="dataset"):
    # Asegurarse de que la carpeta de salida existe
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # Obtener la lista de imágenes en la carpeta de entrada
    archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for archivo in archivos:
        # Extraer la edad del prefijo antes del guion bajo "_"
        edad_match = re.match(r'(\d+)_', archivo)
        if edad_match:
            edad = edad_match.group(1)
            edad_folder = os.path.join(output_base_folder, edad)

            # Crear la carpeta de edad si no existe
            if not os.path.exists(edad_folder):
                os.makedirs(edad_folder)

            # Contador basado en la cantidad de archivos en la carpeta de edad
            contador = len(os.listdir(edad_folder)) + 1
            nuevo_nombre = f"{contador}.jpg"

            # Ruta de archivo original y nueva ruta de archivo
            origen = os.path.join(input_folder, archivo)
            destino = os.path.join(edad_folder, nuevo_nombre)

            # Copiar el archivo a la carpeta de destino
            shutil.copy2(origen, destino)
            print(f"Imagen {archivo} copiada a {destino}")
        else:
            print(f"No se pudo determinar la edad para el archivo {archivo}. Ignorando.")


def contar_imagenes(base_folder="dataset"):
    total_imagenes = 0
    # Recorrer cada subcarpeta (edad) en la carpeta base
    for edad_folder in os.listdir(base_folder):
        edad_path = os.path.join(base_folder, edad_folder)
        # Verificar si es una carpeta
        if os.path.isdir(edad_path):
            # Contar las imágenes en esta subcarpeta
            num_imagenes = len([f for f in os.listdir(edad_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_imagenes += num_imagenes
            print(f"Carpeta {edad_folder}: {num_imagenes} imágenes")

    print(f"Total de imágenes en todas las carpetas: {total_imagenes}")
    return total_imagenes


import os
import shutil


#Para unificar que uno te crea carpetas tipo 1,2,3,4,... y el otro 01,02,03,...
def unificar_carpetas(base_folder="dataset"):
    # Listar todas las carpetas de edad en el directorio base
    carpetas = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # Crear un diccionario para mapear carpetas con y sin ceros iniciales
    mapeo_carpetas = {}

    for carpeta in carpetas:
        # Eliminar ceros iniciales y convertir el nombre en int para comparar
        edad_normalizada = str(int(carpeta))

        if edad_normalizada not in mapeo_carpetas:
            # Agregar carpeta principal sin ceros iniciales
            mapeo_carpetas[edad_normalizada] = carpeta
        else:
            # Si ya existe la carpeta sin ceros, unificar
            carpeta_destino = os.path.join(base_folder, mapeo_carpetas[edad_normalizada])
            carpeta_origen = os.path.join(base_folder, carpeta)

            # Contador para la carpeta de destino
            contador = len(os.listdir(carpeta_destino)) + 1

            for imagen in os.listdir(carpeta_origen):
                if imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Renombrar y mover la imagen
                    nuevo_nombre = f"{contador}.jpg"
                    destino_imagen = os.path.join(carpeta_destino, nuevo_nombre)
                    origen_imagen = os.path.join(carpeta_origen, imagen)

                    # Mover la imagen al destino con el nuevo nombre
                    shutil.move(origen_imagen, destino_imagen)
                    print(f"Imagen {imagen} movida a {destino_imagen}")
                    contador += 1

            # Eliminar la carpeta de origen ya que ahora está vacía
            os.rmdir(carpeta_origen)
            print(f"Carpeta {carpeta} eliminada tras mover sus imágenes a {mapeo_carpetas[edad_normalizada]}")


def copiar_a_rangos(input_folder="dataset", output_folder="dataset_rangos"):
    # Generar rangos automáticamente de 5 en 5 hasta 120 años
    rangos = {}
    for start in range(0, 121, 5):
        end = start + 4
        rango_nombre = f"age_{start}_{end}"
        rangos[rango_nombre] = range(start, end + 1)  # +1 para incluir el último número del rango

    # Crear la carpeta base de rangos si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Recorrer cada carpeta de edad en la carpeta de entrada
    for edad_folder in os.listdir(input_folder):
        edad_path = os.path.join(input_folder, edad_folder)

        # Verificar que sea una carpeta y que el nombre sea un número
        if os.path.isdir(edad_path) and edad_folder.isdigit():
            edad = int(edad_folder)

            # Determinar el rango correspondiente
            rango_destino = None
            for rango_nombre, rango_edad in rangos.items():
                if edad in rango_edad:
                    rango_destino = rango_nombre
                    break

            if rango_destino:
                # Crear la carpeta de rango si no existe
                rango_path = os.path.join(output_folder, rango_destino)
                if not os.path.exists(rango_path):
                    os.makedirs(rango_path)

                # Contador basado en la cantidad de archivos en la carpeta de rango
                contador = len(os.listdir(rango_path)) + 1

                # Copiar cada imagen a la carpeta de rango correspondiente
                for imagen in os.listdir(edad_path):
                    if imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
                        nuevo_nombre = f"{contador}.jpg"
                        origen_imagen = os.path.join(edad_path, imagen)
                        destino_imagen = os.path.join(rango_path, nuevo_nombre)

                        # Copiar la imagen
                        shutil.copy2(origen_imagen, destino_imagen)
                        print(f"Imagen {imagen} copiada a {destino_imagen}")
                        contador += 1
            else:
                print(f"No se encontró un rango para la edad {edad}.")


def organizar_por_edad_AGFW(origen_folder="nuevo_dataset", destino_folder="dataset_rangos"):
    # Crear la carpeta destino si no existe
    if not os.path.exists(destino_folder):
        os.makedirs(destino_folder)

    # Iterar sobre cada carpeta de rango en el dataset de origen
    for rango_folder in os.listdir(origen_folder):
        origen_rango_path = os.path.join(origen_folder, rango_folder)

        # Verificar que sea una carpeta
        if os.path.isdir(origen_rango_path):
            # Ruta de la carpeta de destino
            destino_rango_path = os.path.join(destino_folder, rango_folder)

            # Crear la carpeta de destino si no existe
            if not os.path.exists(destino_rango_path):
                os.makedirs(destino_rango_path)

            # Contador basado en la cantidad de archivos en la carpeta de destino
            contador = len(os.listdir(destino_rango_path)) + 1

            # Copiar cada imagen en la carpeta de rango correspondiente
            for imagen in os.listdir(origen_rango_path):
                if imagen.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Generar nuevo nombre para la imagen basado en el contador
                    nuevo_nombre = f"{contador}.jpg"
                    origen_imagen = os.path.join(origen_rango_path, imagen)
                    destino_imagen = os.path.join(destino_rango_path, nuevo_nombre)

                    # Copiar la imagen al destino con el nuevo nombre
                    shutil.copy2(origen_imagen, destino_imagen)
                    print(f"Imagen {imagen} copiada a {destino_imagen}")
                    contador += 1

if __name__ == "__main__":
    #organizar_por_edad_FGNET('C:/Users/Javier/Desktop/AgeDetection/Piton/data/FGNET/FGNET/images')
    #organizar_por_edad_UTKPlus('C:/Users/Javier/Desktop/AgeDetection/Piton/data/archive/combined_faces/content/combined_faces')
    #unificar_carpetas()
    #copiar_a_rangos()
    #contar_imagenes()
    #organizar_por_edad_AGFW('C:/Users/Javier/Desktop/AgeDetection/Piton/data/AGFW_cropped/cropped/128/female')
    #organizar_por_edad_AGFW('C:/Users/Javier/Desktop/AgeDetection/Piton/data/AGFW_cropped/cropped/128/male')
    contar_imagenes('dataset_rangos')
    print("")
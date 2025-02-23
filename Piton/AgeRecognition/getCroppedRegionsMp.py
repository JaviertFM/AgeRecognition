import warnings
warnings.filterwarnings('ignore')
import os
import tarfile
import json
from time import time
import cv2
import dlib
import numpy as np
from skimage.feature import hog
import pandas as pd
import mediapipe as mp
import math
from tqdm import tqdm
from mediapipe.python.solutions.face_detection import FaceDetection as mpFD
from mediapipe.python.solutions.face_mesh import FaceMesh as mpFM
import sys
import gc

landmarks_mediapipe_dict = {
    "barbilla": [57, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43],
    "bigote": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 432, 436, 426, 327, 328, 2, 97, 98, 206, 216, 212],
    "frente": [54, 68, 9, 298, 284, 332, 297, 338, 10, 109, 67, 103],
    "boca": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 287,
        375, 321, 405, 314, 17, 84, 181, 91, 146, 61
    ],
    "ceja": {
        "derecho": [70, 63, 105, 66, 107, 55, 193],
        "izquierdo": [300, 293, 334, 296, 336, 285, 417],
    },
    "ceno": [107, 9, 336, 285, 417, 168, 193, 55],
    "iris": {
        "derecho": [474, 475, 476, 477],
        "izquierdo": [469, 470, 471, 472],
    },
    "mejilla": {
        "derecho": [234, 227, 123, 207, 57, 150, 136, 172, 58, 132, 93],
        "izquierdo": [454, 447, 352, 436, 379, 365, 397, 367, 288, 435, 361, 401, 323, 366],

    },
    "nariz": {
        "fosas": [49, 64, 98, 99, 94, 328, 327, 294, 279, 4],
        "nariz": [193, 209, 49, 64, 98, 97, 2, 326, 327, 294, 279, 429, 417],
        "tronco": [193, 209, 131, 134, 51, 5, 281, 363, 360, 429, 417]
    },
    "ovalacion_cara": [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338],
    "ojo": {
        "derecho": [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25],
        "izquierdo": [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255],
        },
    "pomulo": {
        "derecho": [116, 111, 117, 118, 119, 120, 121, 47, 126, 142, 36, 205, 187, 123],
        "izquierdo": [345, 340, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352],
    },
    "eje_simetria": {
        "derecho": [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
        "izquierdo":[152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10]
    },
}

landmarks_dlib_dict = {
    "barbilla": [10, 54, 55, 56, 57, 58, 59, 48, 6, 7, 8, 9, 10],
    "bigote": [31, 48, 49, 50, 51, 52, 53, 54, 35, 32, 33, 32],
    "boca": [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60, 48, 59, 58, 57, 56, 55, 54, 64, 63, 62, 61, 60],
    "ceja": {
        "derecha": [17, 18, 19, 20, 21],
        "izquierda": [22, 23, 24, 25, 26]
    },
    "ceno": [21, 22, 42, 39],
    "eje_simetria": {
        "derecho": [27, 8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 18, 19, 20, 21],
        "izquierdo": [27, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22]},
    "iris": {
        "derecho": [37, 38, 40, 41],
        "izquierdo": [43, 44, 46, 47]
    },
    "mejilla": {
        "derecha": [0, 31, 48, 6, 5, 4, 3, 2, 1],
        "izquierda": [16, 35, 54, 10, 11, 12, 13, 14, 15, 16]
    },
    "nariz": {
        "fosas": [30, 31, 32, 33, 34, 35],
        "nariz": [27, 28, 29, 30, 33, 32, 31, 27, 35, 34, 33],
        "tronco": [27, 31, 30, 35],
    },
    "ovalacion_cara": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "pomulo": {
        "derecho": [0, 41, 29, 31],
        "izquierdo": [16, 46, 29, 35]
    },
    "ojo": {
        "derecho": [36, 37, 38, 39, 40, 41],
        "izquierdo": [42, 43, 44, 45, 46, 47]
    },
    "cara_completa": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17],

}

folders_to_create = {
        "brows_left": [],
        "brows_right": [],
        "eyes_left": [],
        "eyes_right": [],
        "BBox": [],
        "BBox_left_eyes": [],
        "BBox_right_eyes": [],
        "frownt" : [],
        "grid": [],
        "hog": [],
        "HSV": [],
        "iris_left": [],
        "iris_right": [],
        "LAB": [],
        "label": [],
        "landmarks": [],
        "mouth": [],
        "excepciones": [],
        "RGB": [],
        "Symmetric_left": [],
        "Symmetric_right": [],
    }

# Cargar el modelo preentrenado de landmarks de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/landmark68")

# Definir las carpetas y parámetros de clasificación para dlib
# regions_indices_dlib = {
#     "eyes": {"left": list(range(36, 42)), "right": list(range(42, 48))},
#     "bb_eyes": {"left": list(range(36, 42)), "right": list(range(42, 48))},
#     "brows": {"left": list(range(17, 22)), "right": list(range(22, 27)), "frownt": [20, 43]},
# }
regions_indices_dlib = {
    "eyes_left": list(range(36, 42)),
    "eyes_right": list(range(42, 48)),
    'BBox_left_eyes': list(range(36, 42)),
    'BBox_right_eyes': list(range(42, 48)),
    "brows_left": list(range(17, 22)),
    "brows_right": list(range(22, 27)),
    "frownt": [20, 43],
}


# Función para convertir la matriz de rotación a ángulos de Euler
def rotation_matrix_to_angles(rotation_matrix):
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi


# Función principal para calcular pitch, roll y yaw
def calculate_pose_angles(landmarks, image):
    h, w = image.shape[:2]

    # Coordenadas 3D de referencia en el mundo real
    face_coordination_in_real_world = np.array([
        [285, 528, 200],  # Landmark 1
        [285, 371, 152],  # Landmark 9
        [197, 574, 128],  # Landmark 57
        [173, 425, 108],  # Landmark 130
        [360, 574, 128],  # Landmark 287
        [391, 425, 108]  # Landmark 359
    ], dtype=np.float64)

    # Extraer las coordenadas 2D de los landmarks relevantes
    face_coordination_in_image = []
    for idx in [1, 9, 57, 130, 287, 359]:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        face_coordination_in_image.append([x, y])

    # Verificar si se obtuvieron todos los landmarks necesarios
    if len(face_coordination_in_image) != 6:
        return None, None, None

    face_coordination_in_image = np.array(face_coordination_in_image, dtype=np.float64)

    # Matriz de la cámara
    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Resolver PnP para obtener el vector de rotación
    success, rotation_vec, _ = cv2.solvePnP(
        face_coordination_in_real_world, face_coordination_in_image,
        cam_matrix, dist_matrix)

    if not success:
        return None, None, None

    # Convertir el vector de rotación a matriz de rotación
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    result = rotation_matrix_to_angles(rotation_matrix)
    pitch, yaw, roll = result[0], result[1], result[2]

    # Limitar los valores entre -90 y 90 grados
    pitch = np.clip(pitch, -90, 90)
    yaw = np.clip(yaw, -90, 90)
    roll = np.clip(roll, -90, 90)

    return pitch, yaw, roll

def project_rotated_square(vertices, angles):
    """
    Rota un cuadrado en 3D según los ángulos dados y proyecta el resultado en el plano XY,
    centrando el cuadrado en el origen (0, 0, 0) antes de rotar, y trasladando el resultado
    a la posición original después de la proyección.

    :param vertices: numpy array de forma (4, 3) con las coordenadas (x, y, z) de los vértices del cuadrado.
    :param angles: Lista o array de tres ángulos en grados para rotación alrededor de los ejes X, Y y Z.
    :return: numpy array de forma (4, 2) con las coordenadas proyectadas y trasladadas a la posición original en el plano XY.
    """
    # Calcular el centro del cuadrado
    center = np.mean(vertices, axis=0)

    # Trasladar el cuadrado al origen
    vertices_translated = vertices - center

    # Convertir los ángulos de grados a radianes
    theta_x, theta_y, theta_z = np.radians(angles)

    # Matriz de rotación alrededor del eje X
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    # Matriz de rotación alrededor del eje Y
    rotation_matrix_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Matriz de rotación alrededor del eje Z
    rotation_matrix_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Aplicar las rotaciones en secuencia: primero X, luego Y, finalmente Z
    rotated_square = vertices_translated @ rotation_matrix_x.T @ rotation_matrix_y.T @ rotation_matrix_z.T

    # Proyectar el cuadrado rotado en el plano XY (ignorar la coordenada Z)
    projected_square = rotated_square[:, :2]

    # Trasladar el cuadrado proyectado a la posición original
    projected_square_original = projected_square + center[:2]

    return projected_square_original


def save_image(image, folder, name):
    """
    Guarda la imagen en la carpeta especificada con el nombre dado.

    :param image: Imagen a guardar.
    :param folder: Carpeta de destino.
    :param name: Nombre del archivo de imagen.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.jpg")
    cv2.imwrite(path, image)

def save_region(img, points, folder):
    """
    Guarda una región específica de la imagen basada en puntos de landmarks.

    :param img: Imagen original.
    :param points: Puntos de landmarks que definen la región.
    :param folder: Carpeta donde se guardará la imagen.
    :param name: Nombre del archivo de imagen.
    """
    x, y, w, h = cv2.boundingRect(np.array(points))
    if 'BBox_left_eyes' in folder or 'BBox_right_eyes' in folder:
        region = img[y - int(w/2):y + int(w/2), x:x + w]
    else:
        region = img[y:y + h, x:x + w]
    return region
def compute_hog_features(image):
    """
    Calcula las características HOG de una imagen.

    :param image: Imagen de entrada.
    :return: Imagen HOG.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, hog_image = hog(gray, visualize=True, block_norm='L2-Hys', feature_vector=False)
    return hog_image

def draw_hog(hog_image):
    """
    Dibuja la representación de las características HOG en un fondo negro.

    :param hog_im: Imagen HOG.
    :return: Imagen con las características HOG dibujadas.
    """
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min()) * 255
    hog_image = hog_image.astype(np.uint8)
    return hog_image


def save_grid(image, projected_square, folder, name):
    """
    Guarda una imagen con un fondo blanco y un polígono negro alrededor del bounding box proyectado.

    :param image: Imagen original.
    :param projected_square: Polígono proyectado como numpy array de forma (4, 2).
    :param folder: Carpeta donde se guardará la imagen.
    :param name: Nombre del archivo de imagen.
    """
    height, width, _ = image.shape
    white_background = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Convertir los puntos del polígono a enteros
    projected_square = np.int32(projected_square)

    # Dibujar el polígono en la imagen
    cv2.fillPoly(white_background, [projected_square], color=(0, 0, 0))

    # Crear la carpeta si no existe
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Guardar la imagen
    save_image(white_background, folder, name)

def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))
def process_image_dlib(image, base_path, name, pitch, yaw, roll, min_width, min_height, regions_to_crop):
    """
    Procesa una imagen para calcular el bounding box, landmarks y regiones de interés,
    y guarda las imágenes en las carpetas correspondientes.

    :param image: Imagen de entrada.
    :param nameTest: Nombre del test.
    :param user: Usuario.
    :param modulo: Módulo utilizado.
    :param screenName: Nombre de la pantalla.
    :param contador_frames: Contador de frames.
    :param TimeDetection: Tiempo de detección.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)
    try:
        #if len(faces) > 0:
        face = max(faces, key=lambda new_face: new_face.width() * new_face.height())
        if (face.top() > 0 and face.bottom() > 0) and (face.left() > 0 and face.right() > 0):
            # Obtener los landmarks
            width = face.width()
            height = face.height()
            if width >= min_width and height >= min_height:
                landmarks = predictor(gray, face)
                lands = [(p.x, p.y) for p in landmarks.parts()]
                if lands:
                    # Guardar la cara detectada
                    centroid = np.mean(np.array(lands)[landmarks_dlib_dict["cara_completa"]], 0)
                    db = max(face.width(), face.height()) // 2
                    frame_height, frame_width = image.shape[:2]
                    bbox_centrado = image[int(max(centroid[1] - db, 0)):int(min(centroid[1] + db, frame_height)),
                                    int(max(centroid[0] - db, 0)):int(min(centroid[0] + db, frame_width)), :]


                    # Guardar HOG de la cara centrada
                    hog_image_o = compute_hog_features(bbox_centrado)
                    black_HOG_o = draw_hog(hog_image_o)

                    # Guardar la representación de mouth
                    nose_tip, chin = lands[30], lands[8]
                    mouth_left, mouth_right = lands[48], lands[54]
                    scrunch_region = image[nose_tip[1]:chin[1], mouth_left[0]:mouth_right[0]]

                    # Guardar la representación de grid
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                    # Convertir el bounding box a un cuadrado
                    width = x2 - x1
                    height = y2 - y1
                    max_side = max(width, height)

                    # Definir los vértices del cuadrado basado en el bounding box
                    vertices = np.array([
                        [x1, y1, 0],  # Punto A
                        [x1 + max_side, y1, 0],  # Punto B
                        [x1 + max_side, y1 + max_side, 0],  # Punto C
                        [x1, y1 + max_side, 0]  # Punto D
                    ])
                    # Obtener el cuadrado proyectado
                    rotation_angles = [pitch, yaw, roll]
                    projected_square = project_rotated_square(vertices, rotation_angles)

                    # Guardar Iris Left
                    lands_iris_l = np.array(lands)[[37, 38, 40, 41],:]
                    lands_iris_l_x = lands_iris_l[:, 0]
                    lands_iris_l_y = lands_iris_l[:, 1]

                    x_min_l, x_max_l = min(lands_iris_l_x), max(lands_iris_l_x)
                    y_min_l, y_max_l = min(lands_iris_l_y), max(lands_iris_l_y)

                    iris_l = image[y_min_l:y_max_l, x_min_l:x_max_l]

                    # Guardar Iris Right
                    lands_iris_r = np.array(lands)[[43, 44, 46, 47], :]
                    lands_iris_r_x = lands_iris_r[:, 0]
                    lands_iris_r_y = lands_iris_r[:, 1]

                    x_min_r, x_max_r = min(lands_iris_r_x), max(lands_iris_r_x)
                    y_min_r, y_max_r = min(lands_iris_r_y), max(lands_iris_r_y)

                    iris_r = image[y_min_r:y_max_r, x_min_r:x_max_r]

                    (mean_lab_image_cropped_adj, mean_hsv_image_cropped_adj,
                     mean_rgb_image_cropped_adj) = (
                        get_lab_hsv_rgb_means(lands, image, regions_to_crop, landmarks_dlib_dict, centroid, db, roll))


                    face_left = crop_face(lands, image, [("eje_simetria", "izquierdo")],landmarks_dlib_dict)

                    rotated_image_left, rotated_centroid_left = reflejar_rotar_pixels_imagen(face_left, np.array(lands)[27,:], np.array(lands)[8,:], centroid, roll)


                    centered_rotated_image_left = rotated_image_left[int(max(rotated_centroid_left[1] - 1.30*db, 0)):int(min(rotated_centroid_left[1] + 1.30*db, frame_height)),
                                    int(max(rotated_centroid_left[0] - 1.30*db, 0)):int(min(rotated_centroid_left[0] + 1.30*db, frame_width)), :]

                    face_right = crop_face(lands, image, [("eje_simetria", "derecho")], landmarks_dlib_dict)
                    rotated_image_right, rotated_centroid_right = reflejar_rotar_pixels_imagen(face_right, np.array(lands)[27, :], np.array(lands)[8, :],
                                                                                   centroid, roll)

                    centered_rotated_image_right = rotated_image_right[int(max(rotated_centroid_right[1] - 1.30*db, 0)):int(
                        min(rotated_centroid_right[1] + 1.30*db, frame_height)),
                                             int(max(rotated_centroid_right[0] - 1.30*db, 0)):int(
                                                 min(rotated_centroid_right[0] + 1.30*db, frame_width)), :]

                    # Guardar las imágenes en LAB y HSV
                    region_list = []
                    region_name = []
                    for region, sides in regions_indices_dlib.items():
                        a_im = save_region(image, [lands[i] for i in sides], f"{base_path}/{region}")
                        region_list.append(a_im)
                        region_name.append(region)

                    if 'nan' in name.lower():
                        is_nan = 1
                    else:
                        is_nan = 0

                    pos = name.split('X')[1].split('_Y')

                    x_bbox = int(max(centroid[0] - db, 0))
                    y_bbox = int(max(centroid[1] - db, 0))
                    width_bbox = int(min(centroid[0] + db, frame_width) - x_bbox)
                    height_bbox = int(min(centroid[1] + db, frame_height) - y_bbox)
                    per = 100 - int(min(x_bbox/frame_width,y_bbox/frame_height,1-(x_bbox+width_bbox)/frame_width,1-(y_bbox+height_bbox)/frame_height)*100)

                    lands_np = np.array(lands)
                    blink_der = distance(lands_np[39],lands_np[36])/max(distance(lands_np[38],lands_np[40]),1)
                    blink_iz = distance(lands_np[45],lands_np[42])/max(distance(lands_np[44],lands_np[46]),1)
                    # Guardar las imágenes en LAB y HSV

                    save_image(iris_r, f"{base_path}/iris_right", f"{name}")
                    save_image(iris_l, f"{base_path}/iris_left", f"{name}")
                    save_grid(image, projected_square, f"{base_path}/grid", f"{name}")
                    save_image(scrunch_region, f"{base_path}/mouth", f"{name}")
                    save_image(black_HOG_o, f"{base_path}/hog", f"{name}")
                    save_image(bbox_centrado, f"{base_path}/BBox", f"{name}")
                    save_image(centered_rotated_image_left, f"{base_path}/Symmetric_left", f"{name}")
                    save_image(centered_rotated_image_right, f"{base_path}/Symmetric_right", f"{name}")
                    # Guardar las imágenes con el promedio por cada canal
                    save_image(mean_lab_image_cropped_adj, f"{base_path}/LAB", f"{name}")
                    save_image(mean_hsv_image_cropped_adj, f"{base_path}/HSV", f"{name}")
                    save_image(mean_rgb_image_cropped_adj, f"{base_path}/RGB", f"{name}")
                    for i in range(len(region_list)):
                        save_image(region_list[i], f"{base_path}/{region_name[i]}", f"{name}")
                    with open(os.path.join(f'{base_path}/landmarks', f'{name}' + '.json'), 'w') as f:
                        json.dump(lands, f)
                    list_data = [name, name[:2], is_nan, pos[0], pos[1], per, x_bbox, y_bbox, height_bbox, width_bbox,
                                 pitch, yaw, roll, blink_der, blink_iz]

                    return list_data
        return None


    except Exception as e:
        print(e)
        save_image(image, f'{base_path}/excepciones', name)
        return None


def reflejar_puntos(puntos, p1, p2):
    # Convertir los puntos de la recta a arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Obtener los parámetros A, B y C de la recta Ax + By + C = 0
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p2[0] * p1[1] - p1[0] * p2[1]

    # Convertir el array de puntos a una matriz
    puntos = np.array(puntos)

    # Calcular el denominador común
    denom = A ** 2 + B ** 2

    # Calcular el punto más cercano en la recta para todos los puntos
    numerador = A * puntos[:, 0] + B * puntos[:, 1] + C
    x_closest = puntos[:, 0] - A * numerador / denom
    y_closest = puntos[:, 1] - B * numerador / denom

    # Calcular el punto reflejado usando el punto más cercano
    x_reflejado = 2 * x_closest - puntos[:, 0]
    y_reflejado = 2 * y_closest - puntos[:, 1]

    # Crear el array de puntos reflejados
    reflejados = np.column_stack((x_reflejado, y_reflejado))

    return np.ceil(reflejados)


def rotar_imagen(image, centroid, angle):
    height, width = image.shape[:2]
    matriz_rotacion = cv2.getRotationMatrix2D(centroid, angle, 1.0)

    image_rotada = cv2.warpAffine(image, matriz_rotacion, (width, height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
    centroid_rotate = np.round(np.dot(matriz_rotacion, np.append(centroid, 1)))
    return image_rotada, centroid_rotate

def slicer_np(a, start, end):
    # Convertir el array a una vista de caracteres individuales
    b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
    # Convertir la vista en una cadena y luego de nuevo a array
    return np.frombuffer(b.tobytes(), dtype=(str, end - start))
def reflejar_rotar_pixels_imagen(new_image, p1, p2, centroid, angle):
    # Inicializar la imagen modificada
    height, width = new_image.shape[:2]
    imagen_reflejada = new_image.copy()

    # Crear una máscara que identifique píxeles que no son 255 en todos los canales
    mask = new_image.sum(axis=-1) < 255 * 3

    # Obtener los índices de los píxeles que satisfacen la condición en la máscara
    indices = np.where(mask)
    puntos = np.array(list(zip(indices[1], indices[0])))  # (x, y) -> (columna, fila)

    # Reflejar los puntos
    puntos_reflejados = reflejar_puntos(puntos, p1, p2)

    # Asegurar que las coordenadas reflejadas estén dentro de los límites de la imagen
    puntos_reflejados = np.clip(puntos_reflejados, [0, 0], [new_image.shape[1] - 1, new_image.shape[0] - 1]).astype(int)

    # Copiar los valores de los píxeles originales a las posiciones reflejadas en el canal actual
    x_orig, y_orig = puntos.T
    x_ref, y_ref = puntos_reflejados.T

    # Asignar valores sin bucles explícitos
    imagen_reflejada[y_ref, x_ref, :] = new_image[y_orig, x_orig, :]

    image_rotada, centroid_rotate = rotar_imagen(imagen_reflejada, centroid, angle)

    return image_rotada, centroid_rotate


def get_lab_hsv_rgb_means(lands, image, regions_to_crop, landmarks_dict, centroid, db, roll):
    img_coppred = crop_face(lands, image, regions_to_crop=regions_to_crop, landmarks_dict = landmarks_dict)

    mask = np.all(img_coppred == [255, 255, 255], axis=-1)
    img_cropped_filtered = img_coppred[~mask]  # Filtrar los píxeles no blancos

    # Convertir la imagen filtrada al espacio de color LAB
    img_filtered_lab = cv2.cvtColor(img_cropped_filtered.reshape((-1, 1, 3)), cv2.COLOR_BGR2LAB)
    img_filtered_hsv = cv2.cvtColor(img_cropped_filtered.reshape((-1, 1, 3)), cv2.COLOR_BGR2HSV)

    # Calcular el valor promedio para cada canal (L, A, B)
    mean_lab = np.mean(img_filtered_lab,0).astype(np.uint8)

    mean_hsv = np.mean(img_filtered_hsv,0).astype(np.uint8)

    mean_rgb = np.mean(img_cropped_filtered,0).astype(np.uint8)

    # Crear imágenes con el promedio para cada canal
    mean_hsv_image = np.full((img_coppred.shape[0], img_coppred.shape[1],3), mean_hsv, dtype=np.uint8)
    mean_lab_image = np.full((img_coppred.shape[0], img_coppred.shape[1],3), mean_lab, dtype=np.uint8)
    mean_rgb_image = np.full((img_coppred.shape[0], img_coppred.shape[1],3), mean_rgb, dtype=np.uint8)


    mean_hsv_image_cropped, new_centroid = rotar_imagen(crop_face(lands, mean_hsv_image, regions_to_crop=["ovalacion_cara"], landmarks_dict = landmarks_dict), centroid, roll)
    mean_lab_image_cropped, new_centroid = rotar_imagen(crop_face(lands, mean_lab_image, regions_to_crop=["ovalacion_cara"], landmarks_dict = landmarks_dict), centroid, roll)
    mean_rgb_image_cropped, new_centroid = rotar_imagen(crop_face(lands, mean_rgb_image, regions_to_crop=["ovalacion_cara"], landmarks_dict = landmarks_dict), centroid, roll)

    frame_height, frame_width = image.shape[:2]

    mean_hsv_image_cropped_adj = mean_hsv_image_cropped[int(max(new_centroid[1] - 1.30*db, 0)):int(
        min(new_centroid[1] + 1.30*db, frame_height)),
                             int(max(new_centroid[0] - 1.30*db, 0)):int(
                                 min(new_centroid[0] + 1.30*db, frame_width)), :]

    mean_lab_image_cropped_adj = mean_lab_image_cropped[int(max(new_centroid[1] - 1.30*db, 0)):int(
            min(new_centroid[1] + 1.30*db, frame_height)),
                                 int(max(new_centroid[0] - 1.30*db, 0)):int(
                                     min(new_centroid[0] + 1.30*db, frame_width)), :]

    mean_rgb_image_cropped_adj = mean_rgb_image_cropped[int(max(new_centroid[1] - 1.30*db, 0)):int(
            min(new_centroid[1] + 1.30*db, frame_height)),
                                 int(max(new_centroid[0] - 1.30*db, 0)):int(
                                     min(new_centroid[0] + 1.30*db, frame_width)), :]


    return (mean_lab_image_cropped_adj, mean_hsv_image_cropped_adj, mean_rgb_image_cropped_adj)

def create_folder(folder_name):
    """
    Crea una carpeta en el directorio actual si no existe.

    :param folder_name: Nombre de la carpeta a crear.
    :return: Ruta absoluta de la carpeta creada.
    """
    try:
        base_path = os.getcwd()
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            # print(f'Carpeta "{folder_name}" creada exitosamente en {folder_path}')
        # else:
            # print(f'La carpeta "{folder_name}" ya existe en {folder_path}')

        return folder_path

    except OSError as e:
        print(f'Error al crear la carpeta "{folder_name}": {e.strerror}')
        return None
    except Exception as e:
        print(f'Ha ocurrido un error inesperado: {str(e)}')
        return None


def extract_folder_from_tar(tar_path, extract_to):
    """
    Extrae todo el contenido de un archivo TAR.GZ en una carpeta específica,
    aplanando la estructura del tar para que el contenido se extraiga directamente.

    :param tar_path: Ruta al archivo TAR.GZ.
    :param extract_to: Ruta donde se desea extraer el contenido.
    """
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            for member in tar_ref.getmembers():
                # Extraer solo los archivos (no carpetas)
                if member.isfile():
                    # Normaliza la ruta del archivo
                    member_path = os.path.normpath(member.name)

                    # Si el archivo está dentro de una carpeta, aplanar la estructura
                    if 'faces' in member_path:
                        # Cambiar la ruta para que se extraiga directamente en extract_to
                        member_path = member_path.replace('faces', 'original_image')
                        # Crear la carpeta si no existe
                        target_path = os.path.join(extract_to, member_path)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        # Extraer el archivo
                        with tar_ref.extractfile(member) as source, open(target_path, 'wb') as dest:
                            dest.write(source.read())
            # print(f'Archivo TAR.GZ "{os.path.basename(tar_path)}" extraído exitosamente en "{extract_to}"')

    except tarfile.ReadError:
        print(f"El archivo '{tar_path}' no es un archivo TAR.GZ válido.")
    except FileNotFoundError:
        print(f"El archivo '{tar_path}' no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al extraer la carpeta: {str(e)}")


def read_config(config_gpt_file):
    """
    Lee el archivo de configuración JSON y extrae los valores necesarios.

    Control de errores agregado para manejar valores incorrectos en el JSON.
    Si los valores no son del tipo esperado, se establecen valores predeterminados.

    :param config_gpt_file: Ruta al archivo JSON de configuración.
    :return: Valores extraídos del archivo de configuración.
    """
    try:
        with open(config_gpt_file, 'r') as file:
            config_gpt = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo {config_gpt_file} no fue encontrado.")
    except json.JSONDecodeError:
        raise ValueError(f"El archivo {config_gpt_file} no contiene un JSON válido.")

    try:
        original_zip_filename = config_gpt['parameter']['zip_filename']
        if not isinstance(original_zip_filename, str):
            raise TypeError("El valor de 'zip_filename' debe ser una cadena de caracteres.")
    except KeyError:
        raise KeyError("Falta la clave 'zip_filename' en la sección 'parameter'.")
    except TypeError as e:
        raise ValueError(f"Error en 'zip_filename': {e}")

    try:
        zip_extraction = config_gpt['parameter']['zip_extraction']
        if not isinstance(zip_extraction, str):
            raise TypeError("El valor de 'zip_extraction' debe ser una cadena de caracteres.")
    except KeyError:
        raise KeyError("Falta la clave 'zip_extraction' en la sección 'parameter'.")
    except TypeError as e:
        raise ValueError(f"Error en 'zip_extraction': {e}")

    pitch_interval = yaw_interval = roll_interval = 90

    filter_name_str = ["B"]

    filter_crop = 100

    min_width = 0
    min_height = 0

    # Intentar leer el valor de "filterNaN"
    filter_nan_value = "no"



    return original_zip_filename, zip_extraction, pitch_interval, yaw_interval, roll_interval, filter_name_str, filter_crop, min_width, min_height, filter_nan_value

def create_sub_folders(base_path, folders_to_create):
    """
        Crea las carpetas y subcarpetas necesarias dentro de una carpeta base.
    """
    for folder, subfolders in folders_to_create.items():
        folder_path = create_folder(os.path.join(base_path, folder))
        for subfolder in subfolders:
            create_folder(os.path.join(folder_path, subfolder))

def create_data_csv(data_path, data):
    path_csv = os.path.join(data_path, 'label', 'features.csv')
    columns = ['Id', 'IdQ', 'NaN', 'x_track', 'y_track', 'roi', 'x_bbox', 'y_bbox', 'height_bbox', 'width_bbox', 'pitch', 'yaw',
                 'roll', 'blink_derecho', 'blink_izquierdo']
    df_data = pd.DataFrame(data, columns=columns)
    df_data.to_csv(path_csv, index=False)

def process_face_mp(image, lands, bbox, base_path, name, pitch, yaw, roll, regions_to_crop):

    try:
        frame_height, frame_width = image.shape[:2]
        lands_adj = np.zeros((lands.shape[0],2))
        lands_adj[:,0] = np.round(lands[:,0]*frame_width)
        lands_adj[:,1] = np.round(lands[:,1]*frame_height)

        #Calculate Frown

        left_X_0 = int(lands_adj[65][0])
        left_X_1 = int(lands_adj[295][0])
        left_Y_0 = int(lands_adj[151][1])
        left_Y_1 = int(lands_adj[197][1])

        frowm = image[left_Y_0:left_Y_1, left_X_0:left_X_1]

        #frown_crop = cv2.resize(frowm, (224, 224), interpolation=cv2.INTER_CUBIC)
        # print('frown', frown_crop.shape)

        #CALCULATE MOUTH

        left_X_0 = int(lands_adj[57][0])
        left_X_1 = int(lands_adj[287][0])
        left_Y_0 = int(lands_adj[164][1])
        left_Y_1 = int(lands_adj[152][1])
        mouth = image[left_Y_0:left_Y_1, left_X_0:left_X_1]
        x_left, y_left = mouth.shape[:2]
        # print('x_left, y_left',x_left, y_left) left_eye_crop = rgb_frame[left_Y_0:left_Y_1, left_X_0:left_X_1]
        #mouth_crop = cv2.resize(mouth, (224, 224), interpolation=cv2.INTER_CUBIC)
        # print('mouth_crop', mouth_crop.shape)



        #Eye recrangle
        left_X_0 = int(lands_adj[362][0])
        left_X_1 = int(lands_adj[263][0])
        left_Y_0 = int(lands_adj[386][1])
        left_Y_1 = int(lands_adj[374][1])

        if left_X_0 == left_X_1:
            left_X_1 = left_X_0+1
        if left_Y_0 == left_Y_1:
            left_Y_1 =left_Y_0+1
        if left_Y_0 > left_Y_1:
            aux = left_Y_1
            left_Y_1 = left_Y_0
            left_Y_0 = aux
        if left_X_0 > left_X_1:
            aux = left_Y_1
            left_X_1 = left_X_0
            left_X_0 = aux

        left_eye_rect = image[left_X_0:left_X_1, left_Y_0:left_Y_1]
        x_left, y_left = left_eye_rect.shape[:2]
        # print('x_left, y_left', x_left, y_left)
        left_eye_crop = image[left_Y_0:left_Y_1, left_X_0:left_X_1]
        #left_eye_crop = cv2.resize(left_eye_crop, (128, 32), interpolation=cv2.INTER_CUBIC)
        # print('left_eye_crop', left_eye_crop.shape)

        right_X_0 = int(lands_adj[33][0])
        right_X_1 = int(lands_adj[173][0])
        right_Y_0 = int(lands_adj[159][1])
        right_Y_1 = int(lands_adj[153][1])
        if right_X_0 == right_X_1:
            right_X_1 = right_X_0+1
        if right_Y_0 == right_Y_1:
            right_Y_1 =right_Y_0+1
        if right_Y_0 > right_Y_1:
            aux = right_Y_1
            right_Y_1 = right_Y_0
            right_Y_0 = aux
        if right_X_0 > right_X_1:
            aux = right_X_1
            right_X_1 = right_X_0
            right_X_0 = aux


        right_eye_rect = image[right_X_0: right_X_1, right_Y_0: right_Y_1]
        x_right, y_right = right_eye_rect.shape[:2]
        # print('x_right, y_right', x_right, y_right)
        right_eye_crop = image[right_Y_0:right_Y_1, right_X_0:right_X_1]
        #right_eye_crop = cv2.resize(right_eye_crop, (128, 32), interpolation=cv2.INTER_CUBIC)
        # print('right_eye_crop', right_eye_crop.shape)


        #EYE BOX

        left_X_0 = int(lands_adj[362][0])
        left_X_1 = int(lands_adj[263][0])
        left_Y_0 = int(lands_adj[295][1])
        left_Y_1 = int(lands_adj[450][1])

        if left_X_0 == left_X_1:
            left_X_1 = left_X_0+1
        if left_Y_0 == left_Y_1:
            left_Y_1 =left_Y_0+1
        if left_Y_0 > left_Y_1:
            aux = left_Y_1
            left_Y_1 = left_Y_0
            left_Y_0 = aux
        if left_X_0 > left_X_1:
            aux = left_Y_1
            left_X_1 = left_X_0
            left_X_0 = aux


        left_eye_rect = image[left_X_0:left_X_1, left_Y_0:left_Y_1]
        x_left, y_left = left_eye_rect.shape[:2]
        # print('x_left, y_left',x_left, y_left)

        left_eye_crop = image[left_Y_0:left_Y_1, left_X_0:left_X_1]
        #left_eye_crop = cv2.resize(left_eye_crop, (224, 224), interpolation=cv2.INTER_CUBIC)

        right_X_0 = int(lands_adj[33][0])
        right_X_1 = int(lands_adj[133][0])
        right_Y_0 = int(lands_adj[65][1])
        right_Y_1 = int(lands_adj[230][1])

        if right_X_0 == right_X_1:
            right_X_1 = right_X_0+1
        if right_Y_0 == right_Y_1:
            right_Y_1 =right_Y_0+1
        if right_Y_0 > right_Y_1:
            aux = right_Y_1
            right_Y_1 = right_Y_0
            right_Y_0 = aux
        if right_X_0 > right_X_1:
            aux = right_X_1
            right_X_1 = right_X_0
            right_X_0 = aux

        right_eye_rect = image[right_X_0: right_X_1, right_Y_0: right_Y_1]
        x_right, y_right = right_eye_rect.shape[:2]
        # print('x_right, y_right',x_right, y_right)
        right_eye_crop = image[right_Y_0:right_Y_1, right_X_0:right_X_1]
        #right_eye_crop = cv2.resize(right_eye_crop, (224, 224), interpolation=cv2.INTER_CUBIC)

        #Calculate Brows
        left_X_0 = int(lands_adj[337][0])
        left_X_1 = int(lands_adj[301][0])
        left_Y_0 = int(lands_adj[337][1])
        left_Y_1 = int(lands_adj[445][1])
        left_brow_rect = image[left_X_0:left_X_1, left_Y_0:left_Y_1]
        x_left, y_left = left_brow_rect.shape[:2]
        # print('x_left, y_left',x_left, y_left)

        if left_X_0 == left_X_1:
            left_X_1 = left_X_0+1
        if left_Y_0 == left_Y_1:
            left_Y_1 =left_Y_0+1
        if left_Y_0 > left_Y_1:
            aux = left_Y_1
            left_Y_1 = left_Y_0
            left_Y_0 = aux
        if left_X_0 > left_X_1:
            aux = left_Y_1
            left_X_1 = left_X_0
            left_X_0 = aux


        left_brow_crop = image[left_Y_0:left_Y_1, left_X_0:left_X_1]
        #left_brow_crop = cv2.resize(left_brow_crop, (128, 32), interpolation=cv2.INTER_CUBIC)

        right_X_0 = int(lands_adj[71][0])
        right_X_1 = int(lands_adj[108][0])
        right_Y_0 = int(lands_adj[108][1])
        right_Y_1 = int(lands_adj[221][1])

        if right_X_0 == right_X_1:
            right_X_1 = right_X_0+1
        if right_Y_0 == right_Y_1:
            right_Y_1 =right_Y_0+1
        if right_Y_0 > right_Y_1:
            aux = right_Y_1
            right_Y_1 = right_Y_0
            right_Y_0 = aux
        if right_X_0 > right_X_1:
            aux = right_X_1
            right_X_1 = right_X_0
            right_X_0 = aux



        right_brow_rect = image[right_X_0: right_X_1, right_Y_0: right_Y_1]
        x_right, y_right = right_brow_rect.shape[:2]
        # print('x_right, y_right',x_right, y_right)
        right_brow_crop = image[right_Y_0:right_Y_1, right_X_0:right_X_1]
        #right_brow_crop = cv2.resize(right_brow_crop, (128, 32), interpolation=cv2.INTER_CUBIC)

        #Compute HOG
        x_min_crop = max(0, int(bbox[1]-bbox[3]))
        x_max_crop = min(frame_width, int(bbox[1] + bbox[3]))
        y_min_crop = max(0, int(bbox[0] - bbox[2]))
        y_max_crop = min(frame_height, int(bbox[0] + bbox[2]))
        #hog_image = compute_hog_features(image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:])
        hog_image = compute_hog_features(image[x_min_crop:x_max_crop, y_min_crop:y_max_crop, :])
        black_HOG_O = draw_hog(hog_image)


        # Guardar la representación de grid
        x1, y1, height, width = bbox

        # Convertir el bounding box a un cuadrado
        max_side = max(width, height)

        # Definir los vértices del cuadrado basado en el bounding box
        vertices = np.array([
            [x1, y1, 0],  # Punto A
            [x1 + max_side, y1, 0],  # Punto B
            [x1 + max_side, y1 + max_side, 0],  # Punto C
            [x1, y1 + max_side, 0]  # Punto D
        ])
        # Obtener el cuadrado proyectado
        rotation_angles = [pitch, yaw, roll]
        projected_square = project_rotated_square(vertices, rotation_angles)
        height_f, width_f, _ = image.shape
        white_background = np.ones((height_f, width_f, 3), dtype=np.uint8) * 255

        # Convertir los puntos del polígono a enteros
        projected_square = np.int32(projected_square)

        # Dibujar el polígono en la imagen
        cv2.fillPoly(white_background, [projected_square], color=(0, 0, 0))

        # Guardar Iris Left
        lands_iris_l = np.array([(int(lands_adj[i][0]), int(lands_adj[i][1])) for i in [469, 470, 471, 472]])
        lands_iris_l_x = lands_iris_l[:, 0]
        lands_iris_l_y = lands_iris_l[:, 1]

        x_min_l, x_max_l = min(lands_iris_l_x), max(lands_iris_l_x)
        y_min_l, y_max_l = min(lands_iris_l_y), max(lands_iris_l_y)

        iris_l = image[y_min_l:y_max_l, x_min_l:x_max_l,:]

        # Guardar Iris Right
        lands_iris_r = np.array([(int(lands_adj[i][0]), int(lands_adj[i][1])) for i in [474, 475, 476, 477]])
        lands_iris_r_x = lands_iris_r[:, 0]
        lands_iris_r_y = lands_iris_r[:, 1]

        x_min_r, x_max_r = min(lands_iris_r_x), max(lands_iris_r_x)
        y_min_r, y_max_r = min(lands_iris_r_y), max(lands_iris_r_y)

        iris_r = image[y_min_r:y_max_r, x_min_r:x_max_r,:]

        centroid = np.mean(np.array(lands_adj)[landmarks_mediapipe_dict["ovalacion_cara"]], 0)
        db = max(width, height) // 2

        (mean_lab_image_cropped_adj, mean_hsv_image_cropped_adj,
         mean_rgb_image_cropped_adj) = (
            get_lab_hsv_rgb_means(lands_adj, image, regions_to_crop, landmarks_mediapipe_dict, centroid, db, roll))




        face_left = crop_face(lands_adj, image, [("eje_simetria", "izquierdo")], landmarks_mediapipe_dict)

        rotated_image_left, rotated_centroid_left = reflejar_rotar_pixels_imagen(face_left, np.array(lands_adj)[10, :],
                                                                                 np.array(lands_adj)[152, :], centroid, roll)

        centered_rotated_image_left = rotated_image_left[int(max(rotated_centroid_left[1] - 1.30*db, 0)):int(
            min(rotated_centroid_left[1] + 1.30*db, frame_height)),
                                      int(max(rotated_centroid_left[0] - 1.30*db, 0)):int(
                                          min(rotated_centroid_left[0] + 1.30*db, frame_width)), :]


        face_right = crop_face(lands_adj, image, [("eje_simetria", "derecho")], landmarks_mediapipe_dict)
        rotated_image_right, rotated_centroid_right = reflejar_rotar_pixels_imagen(face_right, np.array(lands_adj)[10, :],
                                                                                   np.array(lands_adj)[152, :],
                                                                                   centroid, roll)

        centered_rotated_image_right = rotated_image_right[int(max(rotated_centroid_right[1] - 1.30*db, 0)):int(
            min(rotated_centroid_right[1] + 1.30*db, frame_height)),
                                       int(max(rotated_centroid_right[0] - 1.30*db, 0)):int(
                                           min(rotated_centroid_right[0] + 1.30*db, frame_width)), :]

        if 'nan' in name.lower():
            is_nan = 1
        else:
            is_nan = 0

        pos = name.split('X')[1].split('_Y')

        x_bbox = bbox[1]
        y_bbox = bbox[0]
        width_bbox = bbox[3]
        height_bbox = bbox[2]
        per = 100 - int(min(x_bbox / frame_width, y_bbox / frame_height, 1 - (x_bbox + width_bbox) / frame_width,
                            1 - (y_bbox + height_bbox) / frame_height) * 100)

        blink_der = distance(np.array(lands_adj[33]),np.array(lands_adj[133]))/distance(np.array(lands_adj[159]),np.array(lands_adj[145]))
        blink_iz = distance(np.array(lands_adj[463]),np.array(lands_adj[263]))/distance(np.array(lands_adj[386]),np.array(lands_adj[374]))
        list_data = [name, name[:2], is_nan, pos[0], pos[1], per, x_bbox, y_bbox, height_bbox, width_bbox, pitch, yaw,
                     roll, blink_der, blink_iz]
        # Guardar la imagen
        save_image(white_background, f'{base_path}/grid', name)
        save_image(mouth, f'{base_path}/mouth', name)
        save_image(black_HOG_O, f'{base_path}/hog', name)
        save_image(left_brow_crop, f'{base_path}/brows_left', name)
        save_image(right_brow_crop, f'{base_path}/brows_right', name)
        save_image(left_eye_crop, f'{base_path}/BBox_left_eyes', name)
        save_image(right_eye_crop, f'{base_path}/BBox_right_eyes', name)
        save_image(frowm, f'{base_path}/frownt', name)
        save_image(right_eye_crop, f'{base_path}/eyes_right', name)
        save_image(left_eye_crop, f'{base_path}/eyes_left', name)
        save_image(iris_l, f"{base_path}/iris_left", name)
        save_image(iris_r, f"{base_path}/iris_right", name)


        # Guardar las imágenes con el promedio por cada canal

        save_image(mean_lab_image_cropped_adj, f"{base_path}/LAB", f"{name}")

        save_image(mean_hsv_image_cropped_adj, f"{base_path}/HSV", f"{name}")

        save_image(mean_rgb_image_cropped_adj, f"{base_path}/RGB", f"{name}")


        save_image(centered_rotated_image_left, f"{base_path}/Symmetric_left", f"{name}")
        save_image(centered_rotated_image_right, f"{base_path}/Symmetric_right", f"{name}")

        with open(os.path.join(f'{base_path}/landmarks', f'{name}' + '.json'), 'w') as f:
            json.dump(lands.tolist(), f)

        return list_data

    except Exception as e:
        print(f"Excepcion en imagen {name}.")
        print(e)
        save_image(image, f'{base_path}/excepciones', name)
        return None

#

face_mesh_mp = mpFM(static_image_mode=True, max_num_faces=1000, refine_landmarks=True)
def get_landmarks_mp(frame):
    lands = []
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # cap = cv2.VideoCapture(0)

    # print('h', frame_height, 'w', frame_width)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgb_frame = frame.copy()

    rgb_frame.flags.writeable = False
    results = face_mesh_mp.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_face_landmarks:
        if len(results.multi_face_landmarks) == 1:
            lands = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.multi_face_landmarks[0].landmark], dtype=np.float32)

    return lands

face_detector_mp = mpFD(model_selection=0, min_detection_confidence=0.1)


def get_bbox_mp(image, base_path, name, min_width, min_height, lands):
    height, width = image.shape[:2]
    rgb_frame = image.copy()
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bbox_values_centrado = 0
    mp_face_detection = mp.solutions.face_detection
    # cap = cv2.VideoCapture(0)

    fonts = cv2.FONT_HERSHEY_PLAIN
    # ret, frame = cap.read()
    # if ret is False:
    #     break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector_mp.process(frame)

    try:
        face = max(
            results.detections,
            key=lambda deteccion: (deteccion.location_data.relative_bounding_box.width * width *
                                   deteccion.location_data.relative_bounding_box.height * height)
        )
        x = int(face.location_data.relative_bounding_box.xmin * width)
        y = int(face.location_data.relative_bounding_box.ymin * height)
        width = int(face.location_data.relative_bounding_box.width * width)
        height = int(face.location_data.relative_bounding_box.height * height)
        # x, y, w, h = face_react
        # cropped_face = frame[y:width, x:height]


        if (y > min_width and height > min_height) and (x > 0 and width > min_width):
            # cropped_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_CUBIC)
            frame_height, frame_width = image.shape[:2]
            lands_adj = np.zeros((lands.shape[0], 2))
            lands_adj[:, 0] = np.round(lands[:, 0] * frame_width)
            lands_adj[:, 1] = np.round(lands[:, 1] * frame_height)
            centroid = np.mean(np.array(lands_adj)[landmarks_mediapipe_dict["ovalacion_cara"]], 0)
            db = max(height, width) // 2
            #bbox = [x, y, height, width]
            bbox_values_centrado = [int(centroid[0] - db), int(centroid[1] - db), db*2, db*2]
            bbox_centrado = rgb_frame[int(max(centroid[1] - db, 0)):int(min(centroid[1] + db, frame_height)),
                            int(max(centroid[0] - db, 0)):int(min(centroid[0] + db, frame_width)), :]
            save_image(bbox_centrado, f'{base_path}/BBox', name)

    except Exception as e:
        print(e)
        print('img.empty()')


    # print("Returning values: ",bb,lands)
    return bbox_values_centrado


def crop_face(landmarks_points,img, regions_to_crop, landmarks_dict):
    # Diccionario de landmarks
    # Diccionario de landmarks
    height, width = img.shape[:2]

    # Lista de regiones a recortar

    # Crear una imagen en blanco del mismo tamaño que la imagen original
    output_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Procesar cada rostro detectado

    # Obtener los landmarks del rostro detectado

    # Convertir los landmarks en un array de puntos

    for region in regions_to_crop:
        # Si es una región sin especificar el lado (ej. "mejilla" o "ceno")
        if isinstance(region, str):
            # Si la región tiene subregiones (ej. "mejilla", "ojo")
            if region in landmarks_dict and isinstance(landmarks_dict[region], dict):
                # Procesar cada subregión (ej. "derecha", "izquierda")
                for side in landmarks_dict[region]:
                    points = landmarks_dict[region][side]
                    points_array = np.array([landmarks_points[i] for i in points], dtype=np.int32)

                    # Crear una máscara para la subregión
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [points_array], 255)

                    # Invertir la máscara
                    mask_inv = cv2.bitwise_not(mask)

                    # Aplicar la máscara a la imagen original
                    region_cropped = cv2.bitwise_and(img, img, mask=mask)
                    background_white = np.ones_like(img) * 255  # Fondo blanco
                    region_cropped = cv2.add(region_cropped,
                                             cv2.bitwise_and(background_white, background_white, mask=mask_inv))

                    # Crear una imagen en blanco del tamaño de la caja delimitadora de la región
                    x, y, w, h = cv2.boundingRect(points_array)
                    output_image[y:y + h, x:x + w] = region_cropped[y:y + h, x:x + w]

            # Si es una región sin subregiones (ej. "ceno")
            elif region in landmarks_dict:
                points = landmarks_dict[region]
                points_array = np.array([landmarks_points[i] for i in points], dtype=np.int32)

                # Crear una máscara para la región
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points_array], 255)

                # Invertir la máscara
                mask_inv = cv2.bitwise_not(mask)

                # Aplicar la máscara a la imagen original
                region_cropped = cv2.bitwise_and(img, img, mask=mask)
                background_white = np.ones_like(img) * 255  # Fondo blanco
                region_cropped = cv2.add(region_cropped,
                                         cv2.bitwise_and(background_white, background_white, mask=mask_inv))

                # Crear una imagen en blanco del tamaño de la caja delimitadora de la región
                x, y, w, h = cv2.boundingRect(points_array)
                output_image[y:y + h, x:x + w] = region_cropped[y:y + h, x:x + w]

        # Si es una región con un lado específico (ej. ("ojo", "izquierdo"))
        elif isinstance(region, tuple) and region[0] in landmarks_dict:
            region_name, side = region
            region_dict = landmarks_dict[region_name]

            if side in region_dict:
                points = region_dict[side]
                points_array = np.array([landmarks_points[i] for i in points], dtype=np.int32)

                # Crear una máscara para la región
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points_array], 255)

                # Invertir la máscara
                mask_inv = cv2.bitwise_not(mask)

                # Aplicar la máscara a la imagen original
                region_cropped = cv2.bitwise_and(img, img, mask=mask)
                background_white = np.ones_like(img) * 255  # Fondo blanco
                region_cropped = cv2.add(region_cropped,
                                         cv2.bitwise_and(background_white, background_white, mask=mask_inv))

                # Crear una imagen en blanco del tamaño de la caja delimitadora de la región
                x, y, w, h = cv2.boundingRect(points_array)
                output_image[y:y + h, x:x + w] = region_cropped[y:y + h, x:x + w]


    return output_image



def main():
    start_time = time()
    config_file = 'configuracion/config.json'
    config_values = read_config(config_file)

    # Crear carpeta 'dataset'
    dataset_folder = create_folder('dataset')
    if dataset_folder is None:
        print("No se pudo crear la carpeta 'dataset'. El proceso se detendrá.")
        return

    # Extraer valores de configuración
    tar_base_path, zip_extraction, pitch_interval, yaw_interval, roll_interval, filter_names, filter_crop, min_width, min_height, filter_nan_value = config_values

    # Si la extracción está habilitada en la configuración
    if zip_extraction.lower() in ['yes', 'si']:
        extracted_any = False  # Flag para seguir si se extrajo algún archivo

        # Recorrer el directorio base
        for gender in os.listdir(tar_base_path):
            gender_path = os.path.join(tar_base_path, gender)
            if os.path.isdir(gender_path):
                for age in os.listdir(gender_path):
                    age_path = os.path.join(gender_path, age)
                    if os.path.isdir(age_path):
                        for tar_file in os.listdir(age_path):
                            if tar_file.endswith(('.tar.gz', '.tar')):
                                folder_name = os.path.splitext(os.path.splitext(tar_file)[0])[0]
                                extract_to = os.path.join(dataset_folder, gender, age, folder_name)

                                # Verificar si el contenido ya ha sido extraído
                                if not os.path.exists(extract_to):
                                    tar_path = os.path.join(age_path, tar_file)
                                    extract_folder_from_tar(tar_path, os.path.join(dataset_folder, gender, age))
                                    extracted_any = True

        # Si no se extrajo ningún archivo
        if not extracted_any:
            print("Todos los archivos TAR.GZ ya están extraídos en 'dataset'. No se realizó ninguna extracción.")

    sys.stdout.flush()  # Forzar el vaciado del buffer de salida
    print("--------------------- INICIANDO PROCESAMIENTO ---------------------")
    sys.stdout.flush()  # Forzar el vaciado del buffer de salida

    # Definir extensiones válidas para procesamiento
    valid_extensions = ('.jpg', '.jpeg')
    valid_files = []

    # Obtener la lista de subcarpetas (usuarios) en 'dataset' organizado por género y edad
    for gender in os.listdir(dataset_folder):
        if gender == "Hombre":
            regions_to_crop_dlib = ["pomulo", ("nariz", "tronco")]
            regions_to_crop_mp = ["pomulo", ("nariz", "tronco"), "frente"]
        else:
            regions_to_crop_dlib = ["mejilla", "pomulo",  ("nariz", "tronco")]
            regions_to_crop_mp = ["mejilla", "pomulo",  ("nariz", "tronco"), "frente"]
        gender_path = os.path.join(dataset_folder, gender)

        if not os.path.isdir(gender_path):
            continue  # Saltar si no es una carpeta de género

        for age in os.listdir(gender_path):
            age_path = os.path.join(gender_path, age)

            if not os.path.isdir(age_path):
                continue  # Saltar si no es una carpeta de edad

            # Lista de carpetas de usuarios en el directorio actual (dentro de 'gender/age')
            user_folders = [folder_name for folder_name in os.listdir(age_path) if
                            os.path.isdir(os.path.join(age_path, folder_name))]

            with tqdm(total=len(user_folders), desc="Usuarios procesados", position=0, leave=True) as user_bar:
                for user_folder in user_folders:
                    list_data_dlib = []
                    list_data_mp = []
                    user_path = os.path.join(age_path, user_folder)
                    faces_path = os.path.join(user_path, 'faces')
                    subfolder_path = os.path.join(user_path, 'original_image')

                    if os.path.exists(faces_path):
                        os.rename(faces_path, subfolder_path)

                    pyr = []
                    # Verificar si la carpeta original_image existe
                    if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
                        mp_dataset_path = os.path.join(user_path, 'mp_dataset')
                        dlib_dataset_path = os.path.join(user_path, 'dlib_dataset')
                        mp_pyr_path = os.path.join(user_path, 'mp_pyr.csv')

                        if not os.path.exists(mp_pyr_path):
                            create_folder(mp_dataset_path)
                            create_folder(dlib_dataset_path)
                            create_sub_folders(mp_dataset_path, folders_to_create)
                            create_sub_folders(dlib_dataset_path, folders_to_create)

                            # Obtener la lista de archivos válidos en la carpeta original_image
                            # Recorrer los archivos y clasificarlos en válidos e inválidos
                            for f in os.listdir(subfolder_path):
                                file_path = os.path.join(subfolder_path, f)
                                if f.lower().endswith(valid_extensions):
                                    valid_files.append(f)  # Guardar el archivo si tiene una extensión válida
                                else:
                                    os.remove(file_path)


                            # Leer todas las imágenes con extensiones válidas
                            with tqdm(total=len(valid_files), desc=f"Imágenes en {user_folder}", position=1,
                                      leave=False) as img_bar:
                                for file_name in valid_files:
                                    image_path = os.path.join(subfolder_path, file_name)

                                    if filter_nan_value.lower() in ['yes', 'si'] and "nan" in image_path.lower():
                                        continue

                                    try:
                                        image = cv2.imread(image_path)
                                        if image is None:
                                            print(f"Error al leer la imagen en {image_path}. Pasando a la siguiente.")
                                            continue

                                        height_im, width_im, _ = image.shape
                                    except Exception as e:
                                        print(f"Ocurrió un error al procesar la imagen {image_path}: {e}")
                                        continue

                                    if filter_crop < 100:
                                        crop_x = int(width_im * (100 - filter_crop) / 200)
                                        crop_y = int(height_im * (100 - filter_crop) / 200)
                                        image_aux = image[crop_y:height_im - crop_y, crop_x:width_im - crop_x]
                                        gray = cv2.cvtColor(image_aux, cv2.COLOR_RGB2GRAY)
                                        faces_aux = detector(gray)
                                        crop_process = len(faces_aux) > 0
                                    else:
                                        crop_process = True

                                    if crop_process:
                                        lands = get_landmarks_mp(image)
                                        if len(lands) != 0:
                                            pitch, yaw, roll = calculate_pose_angles(lands, image)
                                            pyr.append([os.path.splitext(file_name)[0], pitch, yaw, roll])

                                            if (abs(pitch) < pitch_interval and abs(yaw) < yaw_interval and abs(
                                                    roll) < roll_interval and
                                                    any(filter_name in os.path.splitext(file_name)[0] for filter_name in
                                                        filter_names)):

                                                bbox= get_bbox_mp(image, mp_dataset_path,
                                                                   os.path.splitext(file_name)[0], min_width,
                                                                   min_height, lands)

                                                list_data_d = process_image_dlib(image, dlib_dataset_path,
                                                                   os.path.splitext(file_name)[0], pitch, yaw, roll,
                                                                   min_width, min_height, regions_to_crop_dlib)
                                                if list_data_d != None:
                                                    list_data_dlib.append(list_data_d)
                                                list_data_m = process_face_mp(image, lands, bbox, mp_dataset_path,
                                                                os.path.splitext(file_name)[0], pitch, yaw, roll,
                                                                regions_to_crop_mp)
                                                if list_data_m != None:
                                                    list_data_mp.append(list_data_m)


                                    # Actualizar barra de progreso de imágenes después de procesar cada imagen
                                    img_bar.update(1)

                                # Cerrar la barra de progreso de imágenes después de procesar todas las imágenes
                                img_bar.close()

                            # Guardar los resultados de los ángulos en formato CSV
                            if len(pyr) != 0:
                                df_pyr = pd.DataFrame(pyr, columns=['Id', 'pitch', 'yaw', 'roll'])
                                df_pyr.to_csv(mp_pyr_path, index=False)
                                create_data_csv(dlib_dataset_path,list_data_dlib)
                                create_data_csv(mp_dataset_path,list_data_mp)

                    # Actualizar barra de progreso de usuarios
                    user_bar.update(1)

                    # Liberar memoria
                    gc.collect()

    print(f"Proceso completado en {time() - start_time:.2f} segundos.")


if __name__ == "__main__":
    main()

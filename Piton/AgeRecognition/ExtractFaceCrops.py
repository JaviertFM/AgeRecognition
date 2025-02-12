import os
import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh as mpFM
import mediapipe as mp
import gc

# Define your landmark dictionaries and folder structure for saving crops
landmarks_mediapipe_dict = {
    "barbilla": [57, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43],
    "frente": [54, 68, 9, 298, 284, 332, 297, 338, 10, 109, 67, 103],
    "boca": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 287,
        375, 321, 405, 314, 17, 84, 181, 91, 146, 61
    ],
    "ceja_derecho": [70, 63, 105, 66, 107, 55, 193],
    "ceja_izquierdo": [300, 293, 334, 296, 336, 285, 417],
    "ceno": [107, 9, 336, 285, 417, 168, 193, 55],
    "mejilla_derecho": [234, 227, 123, 207, 57, 150, 136, 172, 58, 132, 93],
    "mejilla_izquierdo": [454, 447, 352, 436, 379, 365, 397, 367, 288, 435, 361, 401, 323, 366],
    "nariz": [193, 209, 49, 64, 98, 97, 2, 326, 327, 294, 279, 429, 417],
    "ovalacion_cara": [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338],
    "ojo_derecho": [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25],
    "ojo_izquierdo": [359, 467, 260, 259, 257, 258, 286, 414, 463, 341, 256, 252, 253, 254, 339, 255],
    "pomulo_derecho": [116, 111, 117, 118, 119, 120, 121, 47, 126, 142, 36, 205, 187, 123],
    "pomulo_izquierdo": [345, 340, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352],
}

def create_face_regions(image, landmarks, regions):
    """
    Takes an image and landmarks, and extracts regions as specified in regions.
    Returns a dictionary with each cropped region.
    """
    regions_dict = {}
    for region_name, points in regions.items():
        # Extract points and define bounding rectangle
        region_points = np.array([landmarks[i] for i in points], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(region_points)
        cropped_region = image[y:y + h, x:x + w]
        regions_dict[region_name] = cropped_region
    return regions_dict

def save_regions(base_folder, age_folder, regions_dict, img_name):
    """
    Saves the cropped regions into corresponding subfolders within the age folder.
    """
    for region_name, cropped_img in regions_dict.items():
        # Define target path for each region
        region_folder = os.path.join(base_folder, age_folder, region_name)
        os.makedirs(region_folder, exist_ok=True)

        # Save image with appropriate naming format
        save_path = os.path.join(region_folder, f"{img_name}_{region_name}.jpg")
        cv2.imwrite(save_path, cropped_img)
        print(f"Saved {region_name} at {save_path}")

def process_images(dataset_folder="dataset_rangos", start_folder="age_25_29"):
    """
    Main processing function: iterates through dataset folders and processes each image,
    starting from the specified folder.
    """
    face_mesh = mpFM(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    start_processing = False  # Flag to check if we reached the starting folder

    for age_folder in os.listdir(dataset_folder):
        age_path = os.path.join(dataset_folder, age_folder)

        # Skip folders until reaching the specified starting folder
        if age_folder == start_folder:
            start_processing = True

        if not start_processing:
            continue

        if os.path.isdir(age_path):
            print(f"Processing folder: {age_folder}")

            # Create "original" folder to store original images
            original_folder = os.path.join(age_path, "original")
            os.makedirs(original_folder, exist_ok=True)

            for img_file in os.listdir(age_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name, _ = os.path.splitext(img_file)
                    img_path = os.path.join(age_path, img_file)

                    # Read and process the image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Could not read image: {img_path}")
                        continue

                    # Detect landmarks using Mediapipe FaceMesh
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_image)

                    # Process only if a face is detected
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0].landmark
                        landmarks = [(int(l.x * image.shape[1]), int(l.y * image.shape[0])) for l in face_landmarks]

                        # Create cropped regions from landmarks
                        regions_dict = create_face_regions(image, landmarks, landmarks_mediapipe_dict)

                        # Verify that all regions have valid crops before saving
                        if all(cropped_img is not None and cropped_img.size > 0 for cropped_img in regions_dict.values()):
                            # Copy the original image to "original" folder since all regions are valid
                            original_save_path = os.path.join(original_folder, img_file)
                            cv2.imwrite(original_save_path, image)
                            print(f"Saved original image to {original_save_path}")

                            # Save each region in its designated folder
                            save_regions(dataset_folder, age_folder, regions_dict, img_name)
                        else:
                            print(f"Incomplete regions for {img_file}, skipping save.")
                    else:
                        print(f"No face detected in {img_file}, skipping region extraction.")

    # Release resources
    face_mesh.close()
    gc.collect()

if __name__ == "__main__":
    process_images()

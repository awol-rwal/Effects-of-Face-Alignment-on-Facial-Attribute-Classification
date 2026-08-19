import cv2
import dlib
import numpy as np
from PIL import Image
import os

# --- Configuration ---
# Point this to the file you downloaded and unzipped
PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat"
DEFAULT_IMAGE_SIZE = (224, 224) # Default size for ResNet, VGG, EfficientNet

# Alter the following values
PADDING = 0.75
SOURCE_PATH = "img_align_celeba"
DESTINATION_PATH = "face_aligned_images"
# ---------------------

# Load Dlib's face detector and landmark predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError as e:
    print(f"Error loading dlib models: {e}")
    print(f"Please make sure '{PREDICTOR_PATH}' is in the correct path.")
    exit()

def align_face(image_path, size=DEFAULT_IMAGE_SIZE, padding=PADDING):
    """
    Detects, aligns, and crops a face from an image.
    
    Args:
        image_path (str): Path to the input image.
        size (tuple): The final (width, height) of the output image.
        padding (float): Percentage to pad the face bounding box.

    Returns:
        PIL.Image or None: An aligned and cropped PIL image, or None if no face is found.
    """
    try:
        # Load image using OpenCV (dlib works well with cv2 images)
        # We read in BGR and convert to RGB for dlib
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read image: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Detect faces
        detections = detector(img_rgb, 1) # Upsample image 1 time

        if len(detections) == 0:
            print(f"No faces found in {image_path}")
            return None

        # 2. Find landmarks for the first (and hopefully only) face
        # We take the largest face if there are multiple
        largest_det = max(detections, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
        shape = predictor(img_rgb, largest_det)

        # 3. Align and crop the face using dlib.get_face_chip
        # This function performs the affine transformation (translation, rotation, scaling)
        # to create a canonical, aligned face.
        face_chip = dlib.get_face_chip(img_rgb, shape, size=max(size))
        
        # Convert the numpy array (from dlib) back to a PIL Image
        return Image.fromarray(face_chip)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

if __name__ == "__main__":
    # Clear destination folder
    directories = [DESTINATION_PATH]
    for directory in directories:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove files
                print(f"{item_path} removed.")

    counter = 1
    faces_not_found = []
    while True:
        try:
            img = cv2.imread(f"{SOURCE_PATH}/{counter:06d}.jpg")
            if img is None:
                print("all images processed")
                break
        except:
            pass
    
        # Align the face to 224x224 (standard for most models)
        aligned_image = align_face(f"{SOURCE_PATH}/{counter:06d}.jpg", size=(224, 224), padding=PADDING)
    
        if aligned_image:
            print(f"{counter:06d}.jpg: Face aligned successfully!")
            aligned_image.save(f"{DESTINATION_PATH}/{counter:06d}.jpg")
        else:
            print("Could not align face.")
            faces_not_found.append(f"{counter:06d}.jpg")
            
        counter += 1

    # Remove faces not found from attributes and partitions
    partition_fobj = open("list_eval_partition.txt", "r")
    labels_fobj = open("list_attr_celeba.txt", "r")
    
    open("face_align_list_partition.txt", "w").close()
    open("face_align_list_attr.txt", "w").close()
    
    face_align_partition_fobj = open("face_align_list_partition.txt", "a")
    face_align_labels_fobj = open("face_align_list_attr.txt", "a")
    
    face_align_labels_fobj.write(labels_fobj.readline())
    face_align_labels_fobj.write(labels_fobj.readline())

    counter = 1
    while True:
        partition_line = partition_fobj.readline()
        labels_line = labels_fobj.readline()
        if not partition_line:
            break
            
        filename = partition_line.strip().split()[0]
        if len(faces_not_found) > 0:
            if filename == faces_not_found[0]:
                faces_not_found.pop(0)
                print(f"{filename} was removed from other text files")
                continue
            
        face_align_partition_fobj.write(partition_line)
        face_align_labels_fobj.write(labels_line)

        counter += 1

    face_align_partition_fobj.close()
    face_align_labels_fobj.close()
    
        


import os
import cv2

from .face_classes import ComponentType, Component

def load_database(database_path) -> dict[ComponentType, list[Component]]:
    components = {'mouths', 'noses', 'eyes', 'brows'}
    cartoon_database = {ComponentType.MOUTH:[], ComponentType.NOSE:[], ComponentType.RIGHT_EYE:[], ComponentType.LEFT_EYE:[], ComponentType.RIGHT_BROW:[], ComponentType.LEFT_BROW:[]}

    for component in components:
        for filename in os.listdir(os.path.join(database_path, component)):
            filepath = os.path.join(database_path, component, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if component == "eyes" or component == "brows":
                component_name = "right_"+component[:-1]
                cartoon_database[ComponentType(component_name)].append((filename, Component(image, [], ComponentType(component_name), (0,0))))

                flipped_image = cv2.flip(image, 1)
                component_name = "left_"+component[:-1]
                cartoon_database[ComponentType(component_name)].append((filename, Component(flipped_image, [], ComponentType(component_name), (0,0))))
            else:
                component_name = component[:-1]
                cartoon_database[ComponentType(component_name)].append((filename, Component(image, [], ComponentType(component_name), (0,0))))
    return cartoon_database

# def load_cartoon_database(database_path):
#     """
#     Load all the images in the provided database directory path.
#     Returns a dictionary keys ["mouth", "left_eye", "right_eye", "left_brow", "right_brow", "nose"] 
#     as provided and each value is a list of multiple numpy.ndarray type objects.
#     """
#
#     cartoon_database = collections.defaultdict(list)
#     for file in os.listdir(os.path.join(database_path, "mouths")):
#         mouth_img = cv2.imread(database_path+"mouths/"+file)
#         cartoon_database[ComponentType.MOUTH].append((file, mouth_img))
#
#     for file in os.listdir(os.path.join(database_path, "noses")):
#         nose_img = cv2.imread(database_path+"noses/"+file)
#         cartoon_database[ComponentType.NOSE].append((file, nose_img))
#
#     for file in os.listdir(os.path.join(database_path, "eyes")):
#         eye_img = cv2.imread(database_path+"eyes/"+file)
#         cartoon_database[ComponentType.RIGHT_EYE].append((file, eye_img))
#         flipped_eye_img = cv2.flip(eye_img, 1)
#         cartoon_database[ComponentType.LEFT_EYE].append((file, flipped_eye_img))
#
#     for file in os.listdir(os.path.join(database_path+"brows")):
#         brow_img = cv2.imread(database_path+"brows/"+file)
#         cartoon_database[ComponentType.RIGHT_BROW].append((file, brow_img))
#         flipped_brow_img = cv2.flip(brow_img, 1)
#         cartoon_database[ComponentType.LEFT_BROW].append((file,flipped_brow_img))
#
#     return cartoon_database



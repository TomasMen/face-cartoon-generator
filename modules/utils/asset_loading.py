import os
import cv2

from .face_classes import ComponentType, Component

def load_database(database_path) -> dict[ComponentType, list[Component]]:
    components = {'mouths', 'noses', 'eyes', 'brows'}
    mouth_labels = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1] 
    cartoon_database = {ComponentType.MOUTH:{"open": [], "closed": []}, ComponentType.NOSE:[], ComponentType.RIGHT_EYE:[], ComponentType.LEFT_EYE:[], ComponentType.RIGHT_BROW:[], ComponentType.LEFT_BROW:[]}

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
                if component_name == "mouth":
                    index = int(filename[:3])
                    if mouth_labels[index] == 0:
                        cartoon_database[ComponentType(component_name)]["closed"].append((filename, Component(image, [], ComponentType(component_name), (0,0))))
                    else:
                        cartoon_database[ComponentType(component_name)]["open"].append((filename, Component(image, [], ComponentType(component_name), (0,0))))
                else:
                    cartoon_database[ComponentType(component_name)].append((filename, Component(image, [], ComponentType(component_name), (0,0))))
    return cartoon_database

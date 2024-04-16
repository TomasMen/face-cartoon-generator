from .utils.face_landmarker import FaceLandmarker
from .utils.face_classes import FaceComposite, Component, ComponentType
from .utils.hair_segmentation import HairSegmentation
from .utils.image_processing import detect_lines_xdog 

import numpy as np

def extract_components(image, face_landmarker: FaceLandmarker, hair_segmentation: HairSegmentation) -> FaceComposite:
    landmark_indexes = {
        ComponentType.MOUTH: [57, 287, 164, 18],
        ComponentType.RIGHT_EYE: [463, 253, 446, 257],
        ComponentType.LEFT_EYE: [243, 23, 226, 27],
        ComponentType.RIGHT_BROW: [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        ComponentType.LEFT_BROW: [107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
        ComponentType.NOSE: [195, 120, 2, 358],
        ComponentType.FACE: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    }

    image_linedrawing = detect_lines_xdog(image)

    landmarks = face_landmarker.detect_landmarks(image)
    if not landmarks:
        raise ValueError("No landmarks were found in the provided image.")

    face_composite = FaceComposite(image, image_linedrawing, landmarks)

    # Extract minimum bounding box around each component
    for component_type in landmark_indexes:

        landmark_coords = [landmarks[x] for x in landmark_indexes[component_type]]

        max_x = max(landmark_coords, key=lambda point: point[0])[0]
        max_y = max(landmark_coords, key=lambda point: point[1])[1]
        min_x = min(landmark_coords, key=lambda point: point[0])[0]
        min_y = min(landmark_coords, key=lambda point: point[1])[1]

        component_linedrawing = image_linedrawing[min_y:max_y, min_x:max_x]

        topleft = (min_x, min_y)
        component = Component(component_linedrawing, landmark_coords, component_type, topleft) 
        face_composite.components[component_type] = component

    hair_mask, hair_label = hair_segmentation.segment_hair(image)
    face_composite.hair_mask = hair_mask
    face_composite.hair_label = hair_label

    mask_white_pixels = np.nonzero(hair_mask == 255)

    min_x = np.min(mask_white_pixels[1])
    min_y = np.min(mask_white_pixels[0])
    max_x = np.max(mask_white_pixels[1])
    max_y = np.max(mask_white_pixels[0])

    topleft = (min_x, min_y)

    hair_image_linedrawing = np.zeros_like(image_linedrawing)
    hair_image_linedrawing[min_y:max_y+1, min_x:max_x+1][hair_mask[min_y:max_y+1, min_x:max_x+1] == 255] = image_linedrawing[min_y:max_y+1, min_x:max_x+1][hair_mask[min_y:max_y+1, min_x:max_x+1] == 255]

    face_composite.components[ComponentType.HAIR] = Component(hair_image_linedrawing, [], ComponentType.HAIR, topleft)

    return face_composite

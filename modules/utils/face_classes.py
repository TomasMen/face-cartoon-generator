from enum import Enum
import numpy as np

class ComponentType(Enum):
    HAIR = "hair"
    LEFT_BROW = "left_brow"
    RIGHT_BROW = "right_brow"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    MOUTH = "mouth"
    NOSE = "nose"
    FACE = "face"

class Component:
    def __init__(self, image, landmarks, component_type, topleft):
        self.image = image
        self.landmarks = landmarks
        self.topleft = topleft
        self.x = topleft[0]
        self.y = topleft[1]
        self.height, self.width = image.shape[:2]
        self.type = component_type
    def set_topleft(self, new_topleft):
        self.topleft = self.x, self.y = new_topleft

class FaceComposite:
    def __init__(self, image, line_image, landmarks):
        self.image = image
        self.line_image = line_image
        self.landmarks = landmarks
        self.hair_mask: np.ndarray
        self.hair_label: str

        self.components = {}

import mediapipe as mp
import cv2
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceLandmarker:
    def __init__(self, model_path):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=False,
                                            output_facial_transformation_matrixes=False,
                                            num_faces=1)
        self.landmark_detector = vision.FaceLandmarker.create_from_options(options)

    def map_normalized_3d_to_image_2d(self, landmarks, image):
        height, width = image.shape[:2]
        image_landmarks = []
        for landmark in landmarks:
            x = landmark.x 
            y = landmark.y

            image_landmarks.append((round(x*width), round(y*height)))

        return image_landmarks

    def detect_landmarks(self, image, map=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmark_detector.detect(mediapipe_image)

        if len(result.face_landmarks) != 0:
            if map:
                return self.map_normalized_3d_to_image_2d(result.face_landmarks[0], image)

            return result.face_landmarks[0]
        else:
            return []

if __name__ == "__main__":
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)

    model_path = os.path.join(current_dir, "..\\..\\data\\face-landmarker-model\\face_landmarker.task")

    landmarker = FaceLandmarker(model_path)

    image_path = os.path.join(current_dir, "..\\..\\data\\test-faces\\level-1\\04-Rosin.png")

    image = cv2.imread(image_path)

    landmarks = landmarker.detect_landmarks(image)

    for landmark in landmarks:
        image[landmark[1], landmark[0], :] = [255, 255, 0] 
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


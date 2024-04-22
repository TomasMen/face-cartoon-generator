import cv2
import numpy as np

from scipy.ndimage import gaussian_filter

K = 1.6
P = 21.7 # Image sharpness
# P = 5.0
PHI = 0.017 # Sharpness of activation falloff, typical range [0.75, 5.0] (essentially how dark the lines are)
EPSILON = 79.5 
SIGMA = 1.4

def detect_lines_fdog(image):
    return image

def detect_lines_xdog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian1 = gaussian_filter(gray, SIGMA)
    gaussian2 = gaussian_filter(gray, K * SIGMA)

    dog = ((1+P) * gaussian1) - (P * gaussian2) 

    xdog = np.zeros_like(dog, dtype=np.float32)

    for i in range(dog.shape[0]):
        for j in range(dog.shape[1]):
            if dog[i, j] >= EPSILON:
                xdog[i,j] = 1.0
            else:
                xdog[i, j] = 1.0 + np.tanh(PHI * (dog[i, j] - EPSILON))

    xdog = (xdog * 255).astype(np.uint8) 

    return xdog


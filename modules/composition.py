import numpy as np
import cv2
import scipy

from .utils.face_classes import FaceComposite, ComponentType

rotation_point_indexes = {
    ComponentType.LEFT_EYE: [33, 243],
    ComponentType.RIGHT_EYE: [463, 263],
    ComponentType.LEFT_BROW: [285, 300],
    ComponentType.RIGHT_BROW: [70, 55],
    ComponentType.MOUTH: [61, 292],
    ComponentType.NOSE: [6, 1],
}

face_contour_landmark_indexes = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def compose_cartoon(face_composite: FaceComposite, matched_cartoon_components):
    canvas = np.ones_like(face_composite.line_image) * 255

    hair_c = face_composite.components[ComponentType.HAIR]
    hair_c.image = cv2.medianBlur(hair_c.image, 5)
    mask_white_pixels = np.nonzero(face_composite.hair_mask == 255)
    canvas[mask_white_pixels] = hair_c.image[mask_white_pixels]

    face_contour_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    thickness = 2
    face_contour_points = [face_composite.landmarks[index] for index in face_contour_landmark_indexes]
    face_contour_points_array = np.array(face_contour_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(face_contour_mask, [face_contour_points_array], isClosed=True, color=(255,), thickness=thickness)

    outline_end_points = [face_composite.landmarks[index] for index in [285, 300, 70, 55]]
    outline_min_y = max(landmark[1] for landmark in outline_end_points)
    face_contour_mask[0:outline_min_y, :] = 0
    canvas[face_contour_mask==255] = 50

    for c_type in matched_cartoon_components:
        component = matched_cartoon_components[c_type][1]

        p1, p2 = (face_composite.landmarks[rotation_point_indexes[c_type][0]],
                  face_composite.landmarks[rotation_point_indexes[c_type][1]])

        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]

        angle_deg = np.degrees(np.arctan2(dy,dx))
        angle_deg = 360 - angle_deg
        angle_deg = angle_deg % 360

        if c_type == ComponentType.NOSE:
            angle_deg += 90
            if angle_deg > 180:
                angle_deg -= 360

        rotated_component_image = scipy.ndimage.rotate(component.image, angle_deg, reshape=True, cval=255)

        canvas[component.y:component.y+rotated_component_image.shape[0], component.x:component.x+rotated_component_image.shape[1]][rotated_component_image!=255] = rotated_component_image[rotated_component_image!=255]
        # canvas[p1[1]-4:p1[1]+5, p1[0]-4:p1[0]+5] = 125
        # canvas[p2[1]-4:p2[1]+5, p2[0]-4:p2[0]+5] = 125

    return canvas


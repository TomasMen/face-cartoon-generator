import cv2
import numpy as np
import math
import random

from .utils.face_classes import Component, ComponentType

def calculate_moments(image) -> list:
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    hu_moments_log = [0.0 for _ in range(len(hu_moments))]
    for i in range(7):
        hu_moments_log[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

    return hu_moments_log

def get_matching_components(face_composite, cartoon_database) -> tuple[dict[ComponentType, tuple[str, Component]], dict[ComponentType, tuple[str, Component]], dict[ComponentType, tuple[str, Component]]]:
    best_matches = {}
    best_matches_symmetric = {}
    random_matches = {}
    component_cutouts = {}
    matched_indexes = {}

    for component_name in cartoon_database:
        max_x, max_y = [max(landmark[index] for landmark in face_composite.components[component_name].landmarks) for index in [0, 1]]
        min_x, min_y = [min(landmark[index] for landmark in face_composite.components[component_name].landmarks) for index in [0, 1]]

        cartoon_components = cartoon_database[component_name]

        component_cutout = face_composite.line_image[min_y:max_y+1, min_x:max_x+1]

        scaled_cartoon_components = []
        for filename, component in cartoon_components:

            ratio = component_cutout.shape[1] / component.image.shape[1]

            target_shape = (round(component.image.shape[1]*ratio), round(component.image.shape[0]*ratio))

            scaled_component_image = cv2.resize(component.image, target_shape)
            scaled_cartoon_components.append((filename, Component(scaled_component_image, [], component.type, component.topleft)))

        max_height = max(component_cutout.shape[0], max(component.image.shape[0] for _, component in scaled_cartoon_components))

        top_padding = 0
        new_top_left = (min_x, min_y)
        if component_cutout.shape[0] < max_height:
            top_padding = max_height-component_cutout.shape[0]
            padded_component_cutout = np.ones((max_height, component_cutout.shape[1]), dtype=np.uint8)*255
            padded_component_cutout[top_padding:, :] = component_cutout 
            new_top_left = (min_x, min_y-top_padding)
            component_cutout = padded_component_cutout

        padded_cartoon_components: list[tuple[str, Component]] = []
        for filename, component in scaled_cartoon_components:
            if component.image.shape[0] < max_height:
                padded_cartoon_component = np.ones((max_height, component_cutout.shape[1]), dtype=np.uint8)*255
                padded_cartoon_component[max_height-component.image.shape[0]:, :] = component.image 
                padded_cartoon_components.append((filename, Component(padded_cartoon_component, component.landmarks, component.type, new_top_left)))
            else:
                component.set_topleft(new_top_left)
                padded_cartoon_components.append((filename, component))

        cutout_moments = calculate_moments(component_cutout)
        cartoon_moments = [calculate_moments(component.image) for _, component in padded_cartoon_components]
        distances = []
        for moments in cartoon_moments:
            moments_arr = np.asarray(moments)
            cutout_moments_arr = np.asarray(cutout_moments)
            distances.append(np.linalg.norm(moments_arr - cutout_moments_arr))

        min_distance = min(distances)
        min_index = distances.index(min_distance)

        symmetric_index = min_index
        matched_indexes[component_name] = min_index
        if component_name == ComponentType.RIGHT_EYE and ComponentType.LEFT_EYE in matched_indexes:
            symmetric_index = matched_indexes[ComponentType.LEFT_EYE]
        elif component_name == ComponentType.LEFT_EYE and ComponentType.RIGHT_EYE in matched_indexes:
            symmetric_index = matched_indexes[ComponentType.RIGHT_EYE]
        elif component_name == ComponentType.RIGHT_BROW and ComponentType.LEFT_BROW in matched_indexes:
            symmetric_index = matched_indexes[ComponentType.LEFT_BROW]
        elif component_name == ComponentType.LEFT_BROW and ComponentType.RIGHT_BROW in matched_indexes:
            symmetric_index = matched_indexes[ComponentType.RIGHT_BROW]

        closest_match = padded_cartoon_components[min_index]

        best_matches[component_name] = closest_match
        best_matches_symmetric[component_name] = padded_cartoon_components[symmetric_index]
        random_matches[component_name] = random.choice(padded_cartoon_components)
        component_cutouts[component_name] = component_cutout

    return best_matches, best_matches_symmetric, random_matches



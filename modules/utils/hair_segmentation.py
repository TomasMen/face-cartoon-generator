import math
import io
import numpy as np
import cv2
import os
import collections
import pickle
import matplotlib.pyplot as plt
import maxflow

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from .face_landmarker import FaceLandmarker
from scipy.ndimage import gaussian_filter

# Interpupillary distance (distance between eyes)
DESIRED_IPD = 100

# Paper implementation parameters
TARGET_THRESHOLD_PERCENTAGE = 0.2
BETA = 0.2 # "Controls balance between detection accuracy and robustness"

class HairSegmentation:
    def __init__(self, training_set_path, landmark_detector=None):
        if not landmark_detector:
            current_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_path)
            model_path = os.path.join(current_dir, "../../data/face-landmarker-model/face_landmarker.task")

            self.landmark_detector = FaceLandmarker(model_path)
        else:
            self.landmark_detector = landmark_detector

        self.training_set_path = training_set_path
        self.hair_likelihood_distribution = { 'short': collections.defaultdict(float), 'long': collections.defaultdict(float) } 
        self.color_distribution = np.zeros((64, 64), dtype=float) 
        
        trainset_directory = os.path.dirname(self.training_set_path)
        hair_likelihood_path = os.path.join(trainset_directory, 'hair_likelihood_distribution.pkl')
        color_distribution_path = os.path.join(trainset_directory, 'color_distribution.pkl')
        if os.path.exists(color_distribution_path) and os.path.exists(hair_likelihood_path):
            with open(hair_likelihood_path, 'rb') as f:
                self.hair_likelihood_distribution = pickle.load(f)
            with open(color_distribution_path, 'rb') as f:
                self.color_distribution = pickle.load(f)
        else:
            self._load_training_data()

    def _load_training_data(self):
        masks_path = os.path.join(self.training_set_path, "masks")
        images_path = os.path.join(self.training_set_path, "pictures")
        output_path = os.path.join(self.training_set_path, "overlays")

        for subdir in os.listdir(masks_path):
            subdir_path = os.path.join(images_path, subdir)
            if not os.path.isdir(subdir_path):
                print(f"Subdirectory {subdir} doesn't exist, skipping")
                continue

            masks_subdir_path = os.path.join(masks_path, subdir)
            output_subdir_path = os.path.join(output_path, subdir)
            os.makedirs(output_subdir_path, exist_ok=True)

            image_count = 0.0
            for filename in os.listdir(subdir_path):
                if not filename.endswith((".jpg", ".png", ".jpeg")):
                    continue

                image_count += 1
                image_path = os.path.join(subdir_path, filename)

                base_file, _ = os.path.splitext(filename)
                mask_path = os.path.join(masks_subdir_path, base_file + ".png")

                image = cv2.imread(image_path)

                landmarks = self.landmark_detector.detect_landmarks(image)
                
                preprocessed_image, new_landmarks, rotation_matrix, scale_factor = self._preprocess_image(image, landmarks)

                if os.path.exists(mask_path):
                    hair_mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                else:
                    print(f"Error: Could not find corresponding hair mask for image {filename}")
                    continue

                hair_mask = np.zeros((hair_mask_image.shape[0], hair_mask_image.shape[1]), dtype=np.uint8)
                hair_mask[hair_mask_image[:, :, 3] > 0] = 255

                mask_height, mask_width = hair_mask_image.shape[:2]
                hair_mask_rotated = cv2.warpAffine(hair_mask, rotation_matrix, (mask_width, mask_height))
                hair_mask_scaled = cv2.resize(hair_mask_rotated, (int(mask_width*scale_factor), int(mask_height*scale_factor)))

                # Print out post-processing mask overlay on the image for process validation
                overlay_image = np.zeros((preprocessed_image.shape[0], preprocessed_image.shape[1], 4), dtype=np.uint8)
                overlay_image[:, :, 3] = hair_mask_scaled * 0.5
                overlay_image[:, :, 0] = 128  # Blue channel of purple color
                overlay_image[:, :, 2] = 128  # Red channel of purple color

                alpha = overlay_image[:, :, 3] / 255.0
                blended_image = cv2.convertScaleAbs(preprocessed_image * (1 - alpha)[:, :, np.newaxis] +
                                                    overlay_image[:, :, :3] * alpha[:, :, np.newaxis])

                output_image_path = os.path.join(output_subdir_path, base_file+'.png')
                cv2.imwrite(output_image_path, blended_image)
                # ---

                hair_pixels = np.argwhere(hair_mask_scaled > 0)

                reference_point = (new_landmarks[473][1] + (new_landmarks[473][1] - new_landmarks[468][1]) // 2, new_landmarks[468][0] + (new_landmarks[473][0] - new_landmarks[468][0]) // 2)

                image_ycrcb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2YCrCb)
                for pixel in hair_pixels:
                    relative_coord = tuple(pixel - reference_point)
                    color_val = np.asarray(image_ycrcb[*relative_coord][1:3])
                    self.hair_likelihood_distribution[subdir][(relative_coord[1], relative_coord[0])] += 1
                    self.color_distribution[*(color_val//4)] += 1

            self.color_distribution /= np.max(self.color_distribution)
            # self.color_distribution = gaussian_filter(self.color_distribution, 1)

            for relative_coord in self.hair_likelihood_distribution[subdir]:
                self.hair_likelihood_distribution[subdir][relative_coord] /= image_count
            
            trainset_directory = os.path.dirname(self.training_set_path)
            hair_likelihood_path = os.path.join(trainset_directory, 'hair_likelihood_distribution.pkl')
            with open(hair_likelihood_path, 'wb') as f:
                pickle.dump(self.hair_likelihood_distribution, f)
            color_distribution_path = os.path.join(trainset_directory, 'color_distribution.pkl')
            with open(color_distribution_path, 'wb') as f:
                pickle.dump(self.color_distribution, f)

    def _preprocess_image(self, image, landmarks):
        normalized_image, normalized_landmarks, rotation_matrix, scale_factor = self._normalize_image(image, landmarks)
        color_corrected_image = self._correct_color_greyworld(normalized_image)

        return color_corrected_image, normalized_landmarks, rotation_matrix, scale_factor

    def _normalize_image(self, image, landmarks):
        eye1 = np.asarray(landmarks[468])
        eye2 = np.asarray(landmarks[473])
        
        dx = eye2[0] - eye1[0]
        dy = eye2[1] - eye1[1]
        current_ipd = np.sqrt(dx**2 + dy**2)

        scale_factor = DESIRED_IPD / current_ipd

        angle_radians = math.atan2(dy, dx)
        angle_degrees = math.degrees(angle_radians)
        
        height, width, _ = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        scaled_image = cv2.resize(rotated_image, (new_width, new_height))

        normalized_landmarks = []
        for landmark in landmarks:
            landmark_centered = np.array([[landmark[0]], [landmark[1]], [1]])
            rotated_landmark_homogenous = np.dot(rotation_matrix, landmark_centered) * scale_factor
            rotated_landmark = [int(rotated_landmark_homogenous[0][0]), int(rotated_landmark_homogenous[1][0])]
            normalized_landmarks.append(rotated_landmark)

        return scaled_image, normalized_landmarks, rotation_matrix, scale_factor

    def _correct_color_greyworld(self, image):
        n_pixels = image.shape[0] * image.shape[1]

        avg_b = np.sum(image[:, :, 0]) / n_pixels
        avg_g = np.sum(image[:, :, 1]) / n_pixels
        avg_r = np.sum(image[:, :, 2]) / n_pixels

        avg_intesity = (avg_b + avg_g + avg_r ) / 3

        illum_b = avg_b / avg_intesity
        illum_g = avg_g / avg_intesity
        illum_r = avg_r / avg_intesity

        corrected_image = np.copy(image)
        corrected_image[:, :, 0] = np.clip(image[:, :, 0] / illum_b, 0, 255).astype(np.uint8)
        corrected_image[:, :, 1] = np.clip(image[:, :, 1] / illum_g, 0, 255).astype(np.uint8)
        corrected_image[:, :, 2] = np.clip(image[:, :, 2] / illum_r, 0, 255).astype(np.uint8)

        return corrected_image

    def _classify_hair_length(self, image, landmarks):
        jaw_p1 = landmarks[172]
        jaw_p2 = landmarks[397]
        jaw_p3 = landmarks[152]
        avg_p1p2_y = ((jaw_p3[1] + jaw_p1[1]) / 2) 

        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = image_greyscale.size

        low = 0
        high = 90
        threshold = 0
        while low <= high:
            mid = (low + high) // 2
            count_thresholded = np.sum(image_greyscale < mid)
            percentage_thresholded = count_thresholded / total_pixels
            
            if percentage_thresholded <= TARGET_THRESHOLD_PERCENTAGE:
                threshold = mid
                low = mid + 1
            else:
                high = mid - 1

        _, image_thresholded = cv2.threshold(image_greyscale, threshold, 1, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        image_opened = cv2.morphologyEx(image_thresholded, cv2.MORPH_OPEN, kernel)
        image_opened[int(avg_p1p2_y):, jaw_p1[0]:jaw_p3[0]+1] = 1 

        vertical_histogram = np.sum(image_opened == 0, axis=1)

        max_hair_y = int((1 - BETA) * jaw_p2[1] + BETA * image.shape[0])

        if np.sum(vertical_histogram[int(avg_p1p2_y):max_hair_y+1]) > 0:
            return "long"
        else:
            return "short"

    def _calculate_average_contrast(self, image):
        luma = image[:, :, 0]
        height, width = image.shape[:2]
        contrasts = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                block = luma[y-1:y+2, x-1:x+2]
                contrast = np.max(block) - np.min(block)
                contrasts.append(contrast)

        return np.mean(contrasts)

    def visualise_distributions(self):
        # Color distribution
        fig, ax = plt.subplots(figsize=(8, 8))

        heatmap = ax.imshow(self.color_distribution, cmap='viridis', aspect='equal')

        cb_ticks = np.arange(0, 64, 4)
        cr_ticks = np.arange(0, 64, 4)[1:3] 
        ax.set_xticks(cb_ticks)
        ax.set_yticks(cr_ticks)
        ax.set_xticklabels(cb_ticks)
        ax.set_yticklabels(cr_ticks)

        ax.set_xlabel('Cb')
        ax.set_ylabel('Cr')
        ax.set_title('Color Distribution Heatmap')

        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label('Probability')

        plt.tight_layout()

        buffer = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buffer)

        buffer.seek(0)
        color_bins_image = plt.imread(buffer, format='png')
        plt.close(fig)

        # Position distributions
        hair_pos_images = {}
        for type in self.hair_likelihood_distribution:
            leftmost = min(x for x, _ in self.hair_likelihood_distribution[type].keys())
            topmost = min(y for _, y in self.hair_likelihood_distribution[type].keys())

            shifted_positions = {(x - leftmost, y - topmost): value for (x, y), value in self.hair_likelihood_distribution[type].items()}

            max_x = max(x for x, _ in shifted_positions.keys())
            max_y = max(y for _, y in shifted_positions.keys())

            hair_pos_image = np.zeros((max_y + 1, max_x + 1), dtype=np.uint8)
            for (x, y), value in shifted_positions.items():
                normalized_value = int(value * 255)
                hair_pos_image[y, x] = normalized_value
            hair_pos_images[type] = hair_pos_image

        return color_bins_image, hair_pos_images

    def segment_hair(self, image, landmarks=None, h_type=None):
        if not landmarks:
            landmarks = self.landmark_detector.detect_landmarks(image)
        
        preprocessed_image, new_landmarks, rotation_matrix, scale_factor = self._preprocess_image(image, landmarks)

        height, width = preprocessed_image.shape[:2]
        graph = maxflow.Graph[float](width*height, 0)

        nodes = graph.add_nodes(height*width)

        hair_type = self._classify_hair_length(preprocessed_image, new_landmarks)
        if h_type:
            hair_type = h_type

        reference_point = np.asarray((new_landmarks[468][0] + (new_landmarks[473][0] - new_landmarks[468][0]) // 2, new_landmarks[468][1] + (new_landmarks[473][1] - new_landmarks[468][1]) // 2))

        preprocessed_image_ycrcb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2YCrCb)
        average_contrast = self._calculate_average_contrast(preprocessed_image_ycrcb)
        # print(f"Average contrast: {average_contrast}")
        # Avoid log(0)
        epsilon = 1e-8
        for y in range(height):
            for x in range(width):
                node_index = y * width + x
                relative_coord = tuple(np.asarray([x, y]) - reference_point)
                if relative_coord in self.hair_likelihood_distribution[hair_type]: 
                    hair_pos_likelihood = self.hair_likelihood_distribution[hair_type][relative_coord]
                else:
                    hair_pos_likelihood = 0
                pixel_color = preprocessed_image_ycrcb[y, x][1:3]
                observed_energy_hair = -1.0 * (np.log(self.color_distribution[*tuple(pixel_color//4)] + epsilon) + np.log(hair_pos_likelihood + epsilon))
                observed_energy_background = -1.0 * (np.log(1 - self.color_distribution[*tuple(pixel_color//4)] + epsilon) + np.log(1-hair_pos_likelihood + epsilon))
                
                graph.add_tedge(nodes[node_index], observed_energy_hair, observed_energy_background)

        for y in range(height):
            for x in range(width):
                node_index = y * width + x
                
                if x < width - 1:
                    neighbor_index = y * width + (x+1)
                    luma_diff = abs(int(preprocessed_image_ycrcb[y, x][0]) - int(preprocessed_image_ycrcb[y, x+1][0]))
                    smoothness = math.exp(-1 * pow(average_contrast+epsilon, -1) * pow(luma_diff,2))
                    # color_dist = np.linalg.norm(preprocessed_image_ycrcb[y, x][1:3] - preprocessed_image_ycrcb[y, x+1][1:])
                    # smoothness = math.exp(-1 *  pow(color_dist,2))
                    graph.add_edge(nodes[node_index], nodes[neighbor_index], smoothness, smoothness)

                if y < height - 1:
                    neighbor_index = (y+1) * width + x
                    luma_diff = abs(int(preprocessed_image_ycrcb[y, x][0]) - int(preprocessed_image_ycrcb[y+1, x][0]))
                    smoothness = math.exp(-1 * pow(average_contrast+epsilon, -1) * pow(luma_diff,2))
                    # color_dist = np.linalg.norm(preprocessed_image_ycrcb[y, x][1:3] - preprocessed_image_ycrcb[y+1, x][1:])
                    # smoothness = math.exp(-1 * pow(color_dist,2))
                    graph.add_edge(nodes[node_index], nodes[neighbor_index], smoothness, smoothness)

        graph.maxflow()
        segmentation = graph.get_grid_segments(nodes)

        hair_indexes = np.where(segmentation)
        hair_coords = []
        for node_index in hair_indexes[0]:
            y = node_index // width
            x = node_index % width
            hair_coords.append((x, y))

        result_hair_mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in hair_coords:
            result_hair_mask[y, x] = 255

        scaled_hair_mask = cv2.resize(result_hair_mask, (image.shape[1], image.shape[0]))
        inverse_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
        rotated_mask = cv2.warpAffine(scaled_hair_mask, inverse_rotation_matrix, (scaled_hair_mask.shape[1], scaled_hair_mask.shape[0]))

        return rotated_mask, hair_type

if __name__ == "__main__":
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    model_path = os.path.join(current_dir, "../../data/face-landmarker-model/face_landmarker.task")
    training_set_path = os.path.join(current_dir, "../../data/segmentation-training-set/")

    landmarker = FaceLandmarker(model_path)
    segmentation = HairSegmentation(training_set_path)

    image_path = os.path.join(current_dir, "../../data/test-faces/level-1/08-Moid-Rasheedi.png")

    image = cv2.imread(image_path)

    landmarks = landmarker.detect_landmarks(image)

    # heatmap, hair_pos_images = segmentation.visualise_distributions()

    # cv2.imwrite("C:\\Users\\yap\\Desktop\\heatmap.png", heatmap)
    # cv2.imwrite("C:\\Users\\yap\\Desktop\\short_hair.png", hair_pos_images["short"])
    # cv2.imwrite("C:\\Users\\yap\\Desktop\\long_hair.png", hair_pos_images["long"])

    kernel = np.ones((3, 3), np.uint8)
    hair_mask = segmentation.segment_hair(image, landmarks, "short")
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel) 

    # print(hair_mask)

    # cv2.imshow("heatmap", heatmap)
    # cv2.imshow("Hair Position", hair_pos_images['short'])

    overlay_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    overlay_image[:, :, 3] = hair_mask * 0.5
    overlay_image[:, :, 1] = 255  # Green channel

    alpha = overlay_image[:, :, 3] / 255.0
    blended_image = cv2.convertScaleAbs(image * (1 - alpha)[:, :, np.newaxis] + overlay_image[:, :, :3] * alpha[:, :, np.newaxis])

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # normalized_image, normalized_landmarks, rotation_matrix, scale_factor = segmentation._preprocess_image(image, landmarks)
    #
    # # for landmark in normalized_landmarks:
    # #     normalized_image[landmark[1], landmark[0], :] = [255, 255, 0] 
    # image[landmarks[468][1], landmarks[468][0], :] = [255, 255, 0] 
    # image[landmarks[473][1], landmarks[473][0], :] = [255, 255, 0] 
    # normalized_image[normalized_landmarks[468][1], normalized_landmarks[468][0], :] = [255, 255, 0] 
    # normalized_image[normalized_landmarks[473][1], normalized_landmarks[473][0], :] = [255, 255, 0] 
    #
    # print(image.shape, normalized_image.shape)
    #
    # if normalized_image.shape[0] > image.shape[0]:
    #     new_image = np.zeros_like(normalized_image)
    #     new_image[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image
    #     image = new_image
    # elif normalized_image.shape[0] < image.shape[0]:
    #     new_image = np.zeros_like(image)
    #     new_image[0:normalized_image.shape[0], 0:normalized_image.shape[1], 0:normalized_image.shape[2]] = normalized_image
    #     normalized_image = new_image
    #
    # print(image.shape, normalized_image.shape)
    #
    # combined_image = cv2.hconcat([image, normalized_image])
    # 
    # cv2.imshow("image", combined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import datetime
import json
import cv2
import os

from modules.extraction import extract_components
from modules.matching import get_matching_components
from modules.composition import compose_cartoon

from modules.utils.face_landmarker import FaceLandmarker
from modules.utils.hair_segmentation import HairSegmentation
from modules.utils.asset_loading import load_database

def main(test_set_path, data_path, output_path, limit_num_images=None):
    test_output_dir = os.path.join(output_path, "full_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    all_images_dir = os.path.join(test_output_dir, "all-images")
    os.makedirs(test_output_dir, exist_ok=False)
    os.makedirs(all_images_dir, exist_ok=False)

    landmarker_model_path = os.path.join(data_path, "face-landmarker-model/face_landmarker.task")
    cartoon_database_path = os.path.join(data_path, "cartoon-database")
    cartoon_database = load_database(cartoon_database_path)

    hair_segmentation_trainset_path = os.path.join(data_path, "segmentation-training-set")

    face_landmarker = FaceLandmarker(landmarker_model_path)
    hair_segmentation = HairSegmentation(hair_segmentation_trainset_path, face_landmarker)

    test_images_dir = os.path.join(test_set_path, "face-images") 
    truth_hairmasks_path = os.path.join(test_set_path, "hair-masks") 

    cartoon_matches_histogram = {}
    for c_type in cartoon_database:
        cartoon_matches_histogram[c_type] = {}
        for filename, _ in cartoon_database[c_type]:
            cartoon_matches_histogram[c_type][filename] = 0
    hairmask_stats = {}

    overall_info_output_path = os.path.join(test_output_dir, "info.json")
    for level_dir in os.listdir(test_images_dir):
        image_level_path = os.path.join(test_images_dir, level_dir)
        masks_level_path = os.path.join(truth_hairmasks_path, level_dir)

        if not os.path.isdir(image_level_path):
            continue

        print(f"Processing {level_dir} :")

        hairmask_stats[level_dir] = {}  

        output_level_path = os.path.join(test_output_dir, level_dir)
        os.makedirs(output_level_path, exist_ok=True)

        hairmask_stats[level_dir]["iou"] = [] 
        hairmask_stats[level_dir]["precision"] = [] 
        hairmask_stats[level_dir]["recall"] = [] 

        img_count = 0
        for image_filename in os.listdir(image_level_path):
            if limit_num_images:
                if img_count > limit_num_images-1:
                    break
                img_count+=1
            if not image_filename.endswith((".jpg", ".png", ".jpeg")):
                continue

            print(f"-> Processing {image_filename}")

            image_path = os.path.join(image_level_path, image_filename)
            base_name = os.path.splitext(image_filename)
            hair_mask_path = os.path.join(masks_level_path, base_name[0]+".png")
            output_image_path = os.path.join(output_level_path, base_name[0])
            os.makedirs(output_image_path, exist_ok=True)

            cartoon_output_path = os.path.join(output_image_path, "final-cartoon.png")
            cartoon_symmetric_output_path = os.path.join(output_image_path, "symmetric.png")
            cartoon_random_output_path = os.path.join(output_image_path, "random.png")

            dirs = [os.path.join(all_images_dir, name) for name in ["normal", "symmetric", "random"]]
            for dir in dirs:
                os.makedirs(dir, exist_ok=True)

            all_images_image_path = os.path.join(all_images_dir, "normal", image_filename)
            all_images_image_path_level = os.path.join(all_images_dir, "normal", level_dir, image_filename)
            os.makedirs(os.path.dirname(all_images_image_path_level), exist_ok=True)
            symmetric_images_image_path = os.path.join(all_images_dir, "symmetric", image_filename)
            symmetric_images_image_path_level = os.path.join(all_images_dir, "symmetric", level_dir, image_filename)
            os.makedirs(os.path.dirname(symmetric_images_image_path_level), exist_ok=True)
            random_images_image_path = os.path.join(all_images_dir, "random", image_filename)
            random_images_image_path_level = os.path.join(all_images_dir, "random", level_dir, image_filename)
            os.makedirs(os.path.dirname(random_images_image_path_level), exist_ok=True)

            image = cv2.imread(image_path)
            truth_hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_UNCHANGED)
            alpha_channel = truth_hair_mask[:, :, 3]
            groundtruth_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

            try:
                face_composite = extract_components(image, face_landmarker, hair_segmentation)
            except ValueError as e:
                print(f"Unable to find landmarks in image {image_path}")
                print(f"Error: {e}")
                continue

            matched_cartoon_components, symmetric_matches, random_matches = get_matching_components(face_composite, cartoon_database) 
            final_cartoon = compose_cartoon(face_composite, matched_cartoon_components)
            final_cartoon_symmetric = compose_cartoon(face_composite, symmetric_matches)
            final_cartoon_random = compose_cartoon(face_composite, random_matches)

            cv2.imwrite(cartoon_output_path, final_cartoon)
            cv2.imwrite(all_images_image_path, final_cartoon)
            cv2.imwrite(all_images_image_path_level, final_cartoon)

            cv2.imwrite(cartoon_symmetric_output_path, final_cartoon_symmetric)
            cv2.imwrite(symmetric_images_image_path, final_cartoon_symmetric)
            cv2.imwrite(symmetric_images_image_path_level, final_cartoon_symmetric)

            cv2.imwrite(cartoon_random_output_path, final_cartoon_random)
            cv2.imwrite(random_images_image_path, final_cartoon_random)
            cv2.imwrite(random_images_image_path_level, final_cartoon_random)

            steps_output_path = os.path.join(output_image_path, "process-steps.jpg")
            original_image_gray = cv2.cvtColor(face_composite.image, cv2.COLOR_BGR2GRAY)
            steps_image = cv2.hconcat([original_image_gray, face_composite.hair_mask, face_composite.line_image, final_cartoon])

            cv2.imwrite(steps_output_path, steps_image)

            info_output_path = os.path.join(output_image_path, "info.json")

            _, groundtruth_mask_binary = cv2.threshold(groundtruth_mask, 127, 255, cv2.THRESH_BINARY)
            _, result_mask_binary = cv2.threshold(face_composite.hair_mask, 127, 255, cv2.THRESH_BINARY)
            intersection = np.logical_and(groundtruth_mask_binary, result_mask_binary)
            union = np.logical_or(groundtruth_mask_binary, result_mask_binary)

            true_positives = np.sum(np.logical_and(groundtruth_mask_binary==255, result_mask_binary==255))
            false_positives = np.sum(np.logical_and(groundtruth_mask_binary==0, result_mask_binary==255))
            false_negatives = np.sum(np.logical_and(groundtruth_mask_binary==255, result_mask_binary==0))
            
            iou = np.sum(intersection) / np.sum(union)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            hairmask_stats[level_dir]["iou"].append(iou) 
            hairmask_stats[level_dir]["precision"].append(precision) 
            hairmask_stats[level_dir]["recall"].append(recall) 

            with open(info_output_path, "w") as info_file:
                info = {}
                info["matches"] = {c_type.value: matched_cartoon_components[c_type][0] for c_type in matched_cartoon_components}
                info["hair_label"] = face_composite.hair_label
                info["iou"] = iou 
                info["precision"] = precision 
                info["recall"] = recall 
                json.dump(info, info_file, indent=4)

            for c_type in matched_cartoon_components:
                matched_filename = matched_cartoon_components[c_type][0]
                cartoon_matches_histogram[c_type][matched_filename] += 1

    # Outside the level loop
    average_level_stats = {}
    for level in hairmask_stats:
        average_level_stats[level] = {}
        average_level_stats[level]["iou"] = np.mean(hairmask_stats[level]["iou"])
        average_level_stats[level]["recall"] = np.mean(hairmask_stats[level]["recall"])
        average_level_stats[level]["precision"] = np.mean(hairmask_stats[level]["precision"])
    average_level_stats["overall"] = {}
    average_level_stats["overall"]["iou"] = np.mean([average_level_stats[level]["iou"] for level in hairmask_stats])
    average_level_stats["overall"]["recall"] = np.mean([average_level_stats[level]["recall"] for level in hairmask_stats])
    average_level_stats["overall"]["precision"] = np.mean([average_level_stats[level]["precision"] for level in hairmask_stats])
    with open(overall_info_output_path, "w") as file:
        info = {}
        info["mask_stats"] = average_level_stats
        info["histogram"] = {c_type.value: cartoon_matches_histogram[c_type] for c_type in cartoon_matches_histogram}
        json.dump(info, file, indent=4)

    create_histogram_image(cartoon_matches_histogram, test_output_dir)

def create_histogram_image(cartoon_matches_histogram, output_dir):
    _, axes = plt.subplots(3, 2, figsize=(15, 10))  # Adjust the figure size as needed
    axes = axes.flatten()  # Flatten the 2D array of axes to simplify indexing

    for i, (component_type, matches) in enumerate(cartoon_matches_histogram.items()):
        labels = matches.keys()
        values = matches.values()
        
        ax = axes[i]
        ax.bar(labels, values, color='skyblue')
        ax.set_title(f"{component_type}")
        ax.set_xlabel('Filename')
        ax.set_ylabel('Count')

        ax.tick_params(labelrotation=45)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.yaxis.grid(True, linestyle='--', linewidth=0.75, color='gray', alpha=0.5)

    plt.tight_layout()
    
    histogram_path = os.path.join(output_dir, 'component_histogram.png')
    plt.savefig(histogram_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for feature extraction.")
    parser.add_argument('test_set_path', type=str, help='Path to the directory containing image levels.')
    parser.add_argument('data_path', type=str, help='Path to the data directory')
    parser.add_argument('--output_path', type=str, help='(optional) Path to the database directory')

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = os.path.join(args.data_path, "..\\output")
        os.makedirs(args.output_path, exist_ok=True)
    
    main(args.test_set_path, args.data_path, args.output_path)

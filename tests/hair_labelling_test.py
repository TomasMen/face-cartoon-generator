ground_truth_labels = {
    "level1": {
        1: "long",
        2: "short",
        3: "short",
        4: "long",
        5: "short",
        6: "short",
        7: "short",
        8: "short",
        9: "short",
        10: "long",
        11: "long",
        12: "short",
        13: "long",
        14: "short",
        15: "long",
        16: "short",
        17: "long",
        18: "long",
        19: "short",
        20: "long",
    },
    "level2": {
        1: "short",
        2: "short",
        3: "long",
        4: "long",
        5: "short",
        6: "long",
        7: "short",
        8: "long",
        9: "short",
        10: "long",
        11: "long",
        12: "long",
        13: "long",
        14: "short",
        15: "long",
        16: "long",
        17: "short",
        18: "short",
        19: "long",
        20: "short",
    },
    "level3": {
        1: "short",
        2: "long",
        3: "long",
        4: "long",
        5: "short",
        6: "short",
        7: "long",
        8: "short",
        9: "short",
        10: "long",
        11: "short",
        12: "long",
        13: "short",
        14: "long",
        15: "long",
        16: "long",
        17: "short",
        18: "short",
        19: "long",
        20: "long",
    }
}

import argparse
import json
import datetime
import cv2
import os

from modules.extraction import extract_components

from modules.utils.face_landmarker import FaceLandmarker
from modules.utils.hair_segmentation import HairSegmentation

def main(test_set_path, data_path, output_path, limit_num_images=None):
    landmarker_model_path = os.path.join(data_path, "face-landmarker-model/face_landmarker.task")

    hair_segmentation_trainset_path = os.path.join(data_path, "segmentation-training-set")

    face_landmarker = FaceLandmarker(landmarker_model_path)
    hair_segmentation = HairSegmentation(hair_segmentation_trainset_path, face_landmarker)

    test_images_dir = os.path.join(test_set_path, "face-images") 

    label_stats = {"correct":[], "correct_count":0, "wrong":[], "wrong_count":0}

    for level_dir in os.listdir(test_images_dir):
        image_level_path = os.path.join(test_images_dir, level_dir)

        if not os.path.isdir(image_level_path):
            continue

        img_count = 0
        level_name = level_dir.replace("-", "")
        for image_filename in os.listdir(image_level_path):
            print(image_filename)
            if limit_num_images:
                if img_count > limit_num_images-1:
                    break
                img_count+=1
            if not image_filename.endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(image_level_path, image_filename)

            image = cv2.imread(image_path)

            num = int(image_filename[:2])

            try:
                face_composite = extract_components(image, face_landmarker, hair_segmentation)
            except ValueError as e:
                print(f"Unable to find landmarks in image {image_path}")
                print(f"Error: {e}")
                continue

            if face_composite.hair_label == ground_truth_labels[level_name][num]:
                label_stats["correct"].append(image_filename)
            else:
                label_stats["wrong"].append(image_filename)
    num_correct = len(label_stats["correct"])
    print(f"Correct labels: {num_correct}:")
    print(label_stats["correct"])
    num_wrong = len(label_stats["wrong"])
    print(f"Wrong labels: {num_wrong}:")
    print(label_stats["wrong"])

    label_stats["wrong_count"] = num_wrong
    label_stats["correct_count"] = num_correct

    info_output = os.path.join(output_path, "hair_labels_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with open(info_output, "w") as file:
        json.dump(label_stats, file)

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

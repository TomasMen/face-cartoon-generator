ground_truth_labels = {
    "level1": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "level2": [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    "level3": [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
}

import argparse
import json
import datetime
import cv2
import os

from modules.utils.face_landmarker import FaceLandmarker

def main(test_set_path, data_path, output_path, limit_num_images=None):
    landmarker_model_path = os.path.join(data_path, "face-landmarker-model/face_landmarker.task")

    face_landmarker = FaceLandmarker(landmarker_model_path)

    test_images_dir = os.path.join(test_set_path, "face-images") 

    label_stats = {"correct_count":0, "wrong_count":0, "correct":[], "wrong":[]}

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

            landmarks = face_landmarker.detect_landmarks(image)
            if not landmarks:
                print(f"Could not find any landmarks in image {image_filename}")
                continue
            mouth_top, mouth_bottom, inner_top, inner_bottom = [landmarks[index] for index in [0, 17, 13, 14]] 
            mouth_height = mouth_bottom[1] - mouth_top[1]
            lip_dist = inner_bottom[1] - inner_top[1] 

            if lip_dist > 0.2 * mouth_height:
                label = 1
            else:
                label = 0

            if label == ground_truth_labels[level_name][num-1]:
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

    info_output = os.path.join(output_path, "mouth_labels_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
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


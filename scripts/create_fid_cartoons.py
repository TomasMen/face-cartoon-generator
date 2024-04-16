import argparse
import sys
import cv2
import os

from modules.utils.image_processing import detect_lines_xdog

def main(dataset_path, output_path):
    print(dataset_path)
    output_dir_path = os.path.join(output_path, "xdog_cartoonset")
    os.makedirs(output_dir_path, exist_ok=False)

    for i, filename in enumerate(os.listdir(dataset_path)):
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            i -= 1
            continue
        if i > 500:
            break
        image_path = os.path.join(dataset_path, filename)
        output_image_path = os.path.join(output_dir_path, filename)

        image = cv2.imread(image_path)
        line_image = detect_lines_xdog(image)

        cv2.imwrite(output_image_path, line_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a black and white cartoon set from colored cartoon images")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory.')
    parser.add_argument('output_path', type=str, help='Path to output the resulting dataset directory.')

    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Invalid dataset path: {args.dataset_path}")
        print("Double check you have inputed the correct path.")
        sys.exit(1)

    main(args.dataset_path, args.output_path)


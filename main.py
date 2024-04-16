import argparse
import datetime
import json
import cv2
import sys
import os

from modules.extraction import extract_components
from modules.matching import get_matching_components
from modules.composition import compose_cartoon
from modules.utils.face_landmarker import FaceLandmarker
from modules.utils.hair_segmentation import HairSegmentation
from modules.utils.asset_loading import load_database
    

def main(image_path, data_path, output_path):
    landmarker_model_path = os.path.join(data_path, "face-landmarker-model/face_landmarker.task")
    cartoon_database_path = os.path.join(data_path, "cartoon-database")
    hair_segmentation_trainset_path = os.path.join(data_path, "segmentation-training-set")

    face_landmarker = FaceLandmarker(landmarker_model_path)
    hair_segmentation = HairSegmentation(hair_segmentation_trainset_path, face_landmarker)

    image = cv2.imread(image_path)
    cartoon_database = load_database(cartoon_database_path)

    try:
        face_composite = extract_components(image, face_landmarker, hair_segmentation)
    except ValueError as e:
        print(f"Unable to find landmarks in image {image_path}")
        print(f"Error: {e}")
        sys.exit(1)

    matched_cartoon_components, _, _ = get_matching_components(face_composite, cartoon_database) 

    final_cartoon = compose_cartoon(face_composite, matched_cartoon_components)

    output_dir = os.path.join(output_path, "main_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=False)

    image_output_path = os.path.join(output_dir, "final-cartoon.jpg")
    cv2.imwrite(image_output_path, final_cartoon)

    steps_output_path = os.path.join(output_dir, "process-steps.jpg")
    original_image_gray = cv2.cvtColor(face_composite.image, cv2.COLOR_BGR2GRAY)
    steps_image = cv2.hconcat([original_image_gray, face_composite.hair_mask, face_composite.line_image, final_cartoon])

    cv2.imwrite(steps_output_path, steps_image)

    info_output_path = os.path.join(output_dir, "info.json")

    with open(info_output_path, "w") as info_file:
        info = {}
        info["matches"] = {c_type.value: matched_cartoon_components[c_type][0] for c_type in matched_cartoon_components}
        info["hair_label"] = face_composite.hair_label
        json.dump(info, info_file, indent=4)

    # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.imshow("Image", final_cartoon)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image for feature extraction.")
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('data_path', type=str, help='Path to the database directory')
    parser.add_argument('--output_path', type=str, help='Path to the database directory')

    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Invalid image path: {args.image_path}")
        print("Double check you have inputed the correct path.")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"Invalid data path: {args.data_path}")
        print("Double check you have inputed the correct path.")
        sys.exit(1)

    if not args.output_path:
        current_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_path)
        args.output_path = os.path.join(current_dir, "output")
        os.makedirs(args.output_path, exist_ok=True)
    
    main(args.image_path, args.data_path, args.output_path)

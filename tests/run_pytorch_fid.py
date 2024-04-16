import os
import json
import sys
import argparse
import datetime
import tempfile

from torchvision import datasets, transforms
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

def main(path_real, path_fake, output_path):
    output_json = os.path.join(output_path, "fid_scores_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    json_dictionary = {"real_set": path_real, "fake_set": path_fake}
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images to fit model input
        transforms.ToTensor(),          # Transform images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained model
    ])

    dataset_real = datasets.ImageFolder(root=path_real, transform=transform)
    dataset_fake = datasets.ImageFolder(root=path_fake, transform=transform)

    tempdir_real = tempfile.mkdtemp()
    tempdir_fake = tempfile.mkdtemp()

    def save_dataset(dataset, directory):
        for i, (img, _) in enumerate(dataset):
            img_path = os.path.join(directory, f'image_{i}.png')
            save_image(img, img_path)

    save_dataset(dataset_real, tempdir_real)
    save_dataset(dataset_fake, tempdir_fake)

    fid_value = calculate_fid_given_paths([tempdir_real, tempdir_fake], 50, 'cpu', 2048)
    json_dictionary["fid_score"] = fid_value
    with open(output_json, "w") as file:
        json.dump(json_dictionary, file)
    print('FID score:', fid_value)

    import shutil
    shutil.rmtree(tempdir_real)
    shutil.rmtree(tempdir_fake)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate fid score between two sets of images.")
    parser.add_argument('truth_set', type=str, help='Path to the truth dataset directory.')
    parser.add_argument('result_set', type=str, help='Path to the resulting dataset directory.')
    parser.add_argument('output_path', type=str, help='Path to output the resulting score.')

    args = parser.parse_args()

    if not os.path.isdir(args.truth_set) or not os.path.isdir(args.result_set):
        print(f"One of the paths provided does not exist!")
        sys.exit(1)

    main(args.truth_set, args.result_set, args.output_path)


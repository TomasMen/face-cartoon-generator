import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from modules.utils.face_classes import ComponentType

def main(json_path, output_path):
    component_ranges = {
        ComponentType.MOUTH: (0, 9),
        ComponentType.NOSE: (0, 29),
        ComponentType.LEFT_EYE: (0, 23),
        ComponentType.RIGHT_EYE: (0, 23),
        ComponentType.LEFT_BROW: (0, 20),
        ComponentType.RIGHT_BROW: (0, 20),
    }
    counts = {}
    for component_type in component_ranges:
        counts[component_type] = {}
        for i in range(component_ranges[component_type][1]+1):
            if i < 10:
                filename = "00"+str(i)+".png" 
            else:
                filename = "0"+str(i)+".png"
            counts[component_type][filename] = 0

    with open(json_path, "r") as jsonfile:
        json_dictionary = json.load(jsonfile)
        print(json_dictionary.keys())

        for component_str in json_dictionary["best_match_histogram"]:
            for filename in json_dictionary["best_match_histogram"][component_str]:
                counts[ComponentType(component_str)][filename] =  json_dictionary["best_match_histogram"][component_str][filename]

    create_histogram_image(counts, output_path)

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
    parser = argparse.ArgumentParser(description="Generate an image with match counts of v1 json format.")
    parser.add_argument('json_path', type=str, help='Path to the json file.')
    parser.add_argument('output_path', type=str, help='Path to output the histogram image.')

    args = parser.parse_args()

    main(args.json_path, args.output_path)

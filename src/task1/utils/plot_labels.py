import pandas as pd
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path

def plot_labeled_regions(image_path, csv_file, figsize=(10, 6), alpha=0.5, show_legend=True):
    """
    Right now, this works with a csv file containing only 1 labelled image, update to be able to process whole file
    Plots an image with overlaid labeled polygon regions from a VIA-style CSV file.

    Parameters:
    - image_path: str, path to the image
    - csv_file: str, path to the VIA-exported CSV containing polygon annotations
    - figsize: tuple, size of the matplotlib figure
    - alpha: float, transparency of the polygons
    - show_legend: bool, whether to show the region IDs in the legend
    """
    # Load CSV and parse shape attributes
    df = pd.read_csv(csv_file)
    df['region_shape_attributes'] = df['region_shape_attributes'].apply(json.loads)
    df['all_points_x'] = df['region_shape_attributes'].apply(lambda x: x.get('all_points_x'))
    df['all_points_y'] = df['region_shape_attributes'].apply(lambda x: x.get('all_points_y'))

    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Plot image and regions
    plt.figure(figsize=figsize)
    plt.imshow(img_np)
    plt.axis('off')

    for _, row in df.iterrows():
        x_coords = row['all_points_x']
        y_coords = row['all_points_y']
        plt.fill(x_coords, y_coords, alpha=alpha, label=f"Region {row['region_id']}")

    if show_legend:
        plt.legend()
    plt.title("Image with Segmented Regions")
    plt.show()
# Example usage:
'''plot_labeled_regions(
    image_path='C:/Users/Gebruiker/Documents/CCS/DLP/Seg_Images/image-data/P21-Fg006-R-C01-R01-binarized.jpg',
    csv_file='C:/Users/Gebruiker/Downloads/dummylabel.csv'
)'''

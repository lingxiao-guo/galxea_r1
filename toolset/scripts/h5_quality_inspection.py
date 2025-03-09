#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""visualize dataset hdf5 file"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
import time
from PIL import Image
import yaml


class GalaxeaDatasetVisualizer:
    def __init__(self, file_path, image_folder=None):
        self.file_path = file_path
        self.file_prefix = file_path.split('.')[0].split('/')[-1]
        self.image_folder = image_folder
        self.width = 320
        self.height = 240
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

   

    def plot_image_data(self):
        """Plot RGB and depth images checking if datasets exist."""

        with h5py.File(self.file_path, 'r') as f:
            # Define image dataset names with the required prefix
            image_datasets = {
                'rgb_head': f'upper_body_observations/rgb_head',
            }

            idx = 0

            fig, ax = plt.subplots(figsize=(10, 8))

            rgb_data = f[image_datasets['rgb_head']][idx]
            np_array = np.frombuffer(rgb_data, np.uint8)
            rgb_image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            rgb_image_resized = cv2.resize(rgb_image, (self.width, self.height))

            ax.imshow(cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB))
            ax.axis('off')  # Hide axes
            ax.set_title(f'RGB Head Image (Index {idx})')

            save_path = f"{self.image_folder}/{self.file_prefix}_head_rgb_image_frame0000.jpg"
            plt.tight_layout()
            plt.savefig(save_path, format='jpg')  # Set quality for JPG (0-100)
            print(f"Saved head image data plot to {save_path}")
            plt.close()


            

# Argument parser for file path
if __name__ == "__main__":
    # load yaml as config
    with open("/workspace/manip-dataset-toolset/manip-dataset-toolset/scripts/h5_path.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["image_path"] = config["h5_path"] + "images_quality_inspection"
    
    time_start = time.time()
    num_count = 0
    for file in os.listdir(config["h5_path"]):
        if file.endswith(".h5"):
            print("inspecting: ", file)
            file_path = os.path.join(config["h5_path"], file)
            visualizer = GalaxeaDatasetVisualizer(file_path, config["image_path"])
            visualizer.plot_image_data()
            num_count += 1
    print(f"Time taken for {num_count}: ", time.time() - time_start)
    # load all the images in imgae_path and generate a gif

    
    
    image_files = sorted([f for f in os.listdir(config["image_path"]) if f.endswith(('.png', '.jpg', '.jpeg'))])
    images = []
    # Load images into the list
    for image_file in image_files:
        image_path = os.path.join(config["image_path"], image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save images as a GIF
    gif_path = config["image_path"] + "/quality_inspection.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=True, duration=500, loop=0)

    print(f"GIF saved to {gif_path}")




                   










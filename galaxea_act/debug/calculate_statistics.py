import os
import h5py
import numpy as np
import cv2
'''
def traverse_h5_keys(group, parent_key=''):
    """
    Recursively traverse all keys in an HDF5 group.
    :param group: The HDF5 group or file to traverse.
    :param parent_key: The parent key path (used for recursion).
    :return: A list of full dataset paths.
    """
    dataset_paths = []
    for key in group.keys():
        full_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(group[key], h5py.Group):  # If it's a group, recurse
            dataset_paths.extend(traverse_h5_keys(group[key], full_key))
        elif isinstance(group[key], h5py.Dataset):  # If it's a dataset, add to list
            dataset_paths.append(full_key)
    return dataset_paths
'''
def calculate_mean_std_from_h5(root_folder):
    # Initialize statistics
    channel_sum = np.zeros(3, dtype=np.float64)  # For R, G, B
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # Datasets to process
    target_datasets = {
        "upper_body_observations/rgb_head",
        "upper_body_observations/rgb_left_hand",
        "upper_body_observations/rgb_right_hand",
    }
    # 存储所有图像的像素值
    all_pixels = []
    # Walk through the directory tree
    for subdir, _, files in os.walk(root_folder):
        print("subdir: ", subdir)
        for file in files:
            if file.endswith('.h5'):  # Check if the file is an HDF5 file
                # print("file: ",file)
                h5_path = os.path.join(subdir, file)
                
                # Open the HDF5 file
                with h5py.File(h5_path, 'r') as h5_file:
                    # Get all dataset paths
                    # dataset_paths = traverse_h5_keys(h5_file)
                    
                    # Process only target datasets
                    for dataset_path in target_datasets:
                            compressed_images = h5_file[dataset_path][...]  # Read the compressed images
                            
                            # Iterate through the compressed images
                            for compressed_image in compressed_images:
                                # Decompress the image (example assumes OpenCV-compatible decompression)
                                image = cv2.imdecode(np.frombuffer(compressed_image, np.uint8), cv2.IMREAD_COLOR)
                                
                                if image is None:
                                    continue  # Skip if decompression fails

                                # Convert BGR to RGB (if OpenCV is used)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
                                image = cv2.resize(image, (240, 320))
                                
                                # Convert to float32 to avoid overflow
                                image_float = image.astype(np.float32)

                                # Update statistics
                                channel_sum += image_float.sum(axis=(0, 1))
                                channel_squared_sum += (image_float ** 2).sum(axis=(0, 1))
                                total_pixels += image.shape[0] * image.shape[1]
                                
                                #all_pixels.extend(image.reshape(-1, 3))
                                #print("IMAGE.SHAPE:", image.shape, len(all_pixels))
                                # Update statistics
                                #total_pixels += image.shape[0] * image.shape[1]
                                #channel_sum += image.sum(axis=(0, 1))
                                
                                #channel_squared_sum += (image ** 2).sum(axis=(0, 1))
                                #print("image:",image,"image2:",image**2)

    total_pixels = float(total_pixels)
    # Calculate mean and standard deviation
    mean = channel_sum / total_pixels
    std = np.sqrt(channel_squared_sum / total_pixels - mean ** 2)
    #mean_all_pixels = np.mean(all_pixels, axis=0)
    #variance_all_pixels = np.mean((all_pixels - mean_all_pixels) ** 2, axis=0)
    # 计算标准差
    #std_dev_all_pixels = np.sqrt(variance_all_pixels)
    mean/=255.0
    std/=255.0
    return mean, std  # mean_all_pixels, std_dev_all_pixels  # 

# Example usage
root_folder = "/data/GalaxeaDatasetH5/TrajectoryDataH5"  # "/data/my_cool_dataset/r1_data"
mean, std = calculate_mean_std_from_h5(root_folder)
print("Mean:", mean)
print("Standard Deviation:", std)
np.save('mean.npy', mean)
np.save('std.npy', std)

# Save to .txt files
with open('mean.txt', 'w') as f:
    f.write('Mean:\n')
    f.write(', '.join(map(str, mean)))

with open('std.txt', 'w') as f:
    f.write('Standard Deviation:\n')
    f.write(', '.join(map(str, std)))

print("Saved mean and std to 'mean.npy', 'std.npy', 'mean.txt', and 'std.txt'")
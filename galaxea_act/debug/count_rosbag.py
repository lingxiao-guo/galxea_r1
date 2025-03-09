import os
import sys

def count_rosbags(path):
    rosbag_count = 0
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        # Count the number of .bag files in each directory
        rosbag_count += sum(1 for file in files if file.endswith('.bag'))
    
    return rosbag_count

# Replace with your target directory path
directory_path = sys.argv[1]
rosbag_count = count_rosbags(directory_path)

print(f"Total number of rosbags: {rosbag_count}")
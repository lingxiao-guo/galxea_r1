#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""visualize dataset hdf5 file"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

class GalaxeaDatasetVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_prefix = file_path.split('.')[0]

    def plot_arm_data(self, side='left'):
        """Plot arm joint positions, velocities, gripper positions, and EE poses."""
        assert side in ['left', 'right'], "side must be 'left' or 'right'"

        datasets = {
            'joint_position': f'upper_body_observations/{side}_arm_joint_position',
            'joint_velocity': f'upper_body_observations/{side}_arm_joint_velocity',
            'gripper_position': f'upper_body_observations/{side}_arm_gripper_position',
        }

        with h5py.File(self.file_path, 'r') as f:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Joint position plot
            if len(f[datasets['joint_position']]) > 0:
                joint_position_data = f[datasets['joint_position']][:]
                for i in range(joint_position_data.shape[1]):
                    axs[0].plot(joint_position_data[:, i], label=f'Joint {i+1}')
                axs[0].set_title(f'{side.capitalize()} Arm Joint Positions')
                axs[0].legend()
            else:
                axs[0].axis('off')

            # Joint velocity plot
            if len(f[datasets['joint_velocity']]) > 0:
                joint_velocity_data = f[datasets['joint_velocity']][:]
                for i in range(joint_velocity_data.shape[1]):
                    axs[1].plot(joint_velocity_data[:, i], label=f'Velocity {i+1}')
                axs[1].set_title(f'{side.capitalize()} Arm Joint Velocities')
                axs[1].legend()
            else:
                axs[1].axis('off')

            # Gripper position plot
            if len(f[datasets['gripper_position']]) > 0:
                gripper_position_data = f[datasets['gripper_position']][:]
                axs[2].plot(gripper_position_data[:, 0], label='Gripper Position')
                axs[2].set_title(f'{side.capitalize()} Gripper Position')
                axs[2].legend()
            else:
                axs[2].axis('off')

            plt.tight_layout()

            # Save the plot
            save_path = f"{self.file_prefix}_{side}_arm_data.png"
            plt.savefig(save_path)
            print(f"Saved {side} arm data plot to {save_path}")
            plt.close()

    def plot_command_data(self, side='left'):
        """Plot command data for the arm, including EE pose, joint position, and gripper position."""
        assert side in ['left', 'right'], "side must be 'left' or 'right'"

        datasets = {
            'ee_pose_cmd': f'upper_body_action_dict/{side}_arm_ee_pose_cmd',
            #'joint_position_cmd': f'upper_body_action_dict/{side}_arm_joint_position_cmd',
            'gripper_position_cmd': f'upper_body_action_dict/{side}_arm_gripper_position_cmd'
        }

        with h5py.File(self.file_path, 'r') as f:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # EE pose command plot
            if len(f[datasets['ee_pose_cmd']]) > 0:
                ee_pose_cmd_data = f[datasets['ee_pose_cmd']][:, :3]
                for i in range(ee_pose_cmd_data.shape[1]):
                    axs[0].plot(ee_pose_cmd_data[:, i], label=f'EE Axis {i+1}')
                axs[0].set_title(f'{side.capitalize()} EE Pose Command (First 3 Axes)')
                axs[0].legend()
            else:
                axs[0].axis('off')
            '''
            # Joint position command plot
            if len(f[datasets['joint_position_cmd']]) > 0:
                joint_position_cmd_data = f[datasets['joint_position_cmd']][:]
                axs[1].plot(joint_position_cmd_data, label=f'Joint Position Command')
                axs[1].set_title(f'{side.capitalize()} Joint Position Command')
                axs[1].legend()
            else:
                axs[1].axis('off')
            '''
            # Gripper position command plot
            if len(f[datasets['gripper_position_cmd']]) > 0:
                gripper_position_cmd_data = f[datasets['gripper_position_cmd']][:]
                axs[2].plot(gripper_position_cmd_data[:, 0], label='Gripper Command')
                axs[2].set_title(f'{side.capitalize()} Gripper Position Command')
                axs[2].legend()
            else:
                axs[2].axis('off')

            plt.tight_layout()

            # Save the plot
            save_path = f"{self.file_prefix}_{side}_arm_command_data.png"
            plt.savefig(save_path)
            print(f"Saved {side} arm command data plot to {save_path}")
            plt.close()

    def plot_image_data(self):
        """Plot RGB and depth images checking if datasets exist."""

        with h5py.File(self.file_path, 'r') as f:
            # Define image dataset names with the required prefix
            image_datasets = {
                'rgb_head': f'upper_body_observations/rgb_head',
                'rgb_left_hand': f'upper_body_observations/rgb_left_hand',
                'rgb_right_hand': f'upper_body_observations/rgb_right_hand',
                'depth_head': f'upper_body_observations/depth_head',
                'depth_left_hand': f'upper_body_observations/depth_left_hand',
                'depth_right_hand': f'upper_body_observations/depth_right_hand',
            }

            # Get evenly spaced indices
            num_images = f[image_datasets['rgb_head']].shape[0]
            indices = np.linspace(0, num_images - 1, num=5, dtype=int)

            fig, axs = plt.subplots(3, 5, figsize=(20, 12))
            # Plot RGB images
            for i, idx in enumerate(indices):
                for row, arm in enumerate(['head', 'left_hand', 'right_hand']):
                    dataset_name = f'rgb_{arm}'
                    if dataset_name in f['upper_body_observations']:
                        rgb_data = f[image_datasets[f'rgb_{arm}']][idx]
                        np_array = np.frombuffer(rgb_data, np.uint8)
                        rgb_image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
                        rgb_image_resized = cv2.resize(rgb_image, (320, 240))
                        axs[row, i].imshow(cv2.cvtColor(rgb_image_resized, cv2.COLOR_BGR2RGB))
                        axs[row, i].axis('off')  # Hide axes
                    else:
                        axs[row, i].axis('off')  # Leave blank if dataset is missing
                    axs[row, i].set_title(f'RGB {arm} Image {i + 1} (Index {idx})')

            # Save image data plots
            save_path = f"{self.file_prefix}_arm_rgb_image.png"
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Saved image data plot for rgb arm to {save_path}")
            plt.close()

            # Plot Depth images
            fig, axs = plt.subplots(3, 5, figsize=(20, 12))
            for i, idx in enumerate(indices):
                for row, arm in enumerate(['head', 'left_hand', 'right_hand']):
                    dataset_name = f'depth_{arm}'
                    if dataset_name in f['upper_body_observations']:
                        depth_data = f[image_datasets[f'depth_{arm}']][idx]
                        np_array = np.frombuffer(depth_data, np.uint8)
                        depth_image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
                        depth_image_resized = cv2.resize(depth_image, (320, 240))

                        # Normalize depth image and convert it to grayscale
                        depth_image_normalized = np.clip(depth_image_resized / 6000, 0, 1)
                        depth_image_normalized = (depth_image_normalized * 255).astype(np.uint8)

                        axs[row, i].imshow(depth_image_normalized, cmap='gray')
                        axs[row, i].axis('off')  # Hide axes
                    else:
                        axs[row, i].axis('off')  # Leave blank if dataset is missing
                    axs[row, i].set_title(f'Depth {arm} Image {i + 1} (Index {idx})')
            save_path = f"{self.file_prefix}_arm_depth_image.png"
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Saved image data plot for rgb arm to {save_path}")
            plt.close()
            

# Argument parser for file path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HDF5 Dataset")
    parser.add_argument("file_path", type=str, help="Path to the HDF5 file")
    args = parser.parse_args()

    # Create visualizer instance and plot both arms
    visualizer = GalaxeaDatasetVisualizer(args.file_path)
    visualizer.plot_arm_data(side='left')
    visualizer.plot_arm_data(side='right')
    visualizer.plot_command_data(side='left')
    visualizer.plot_command_data(side='right')
    visualizer.plot_image_data()




                   










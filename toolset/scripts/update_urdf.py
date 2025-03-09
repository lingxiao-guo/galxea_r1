import os
import sys
import h5py
import glob
import tqdm
import numpy as np
import pyquaternion as pyQuaternion
import manip_dataset_toolset.utlis.kdl_utils as utlis

np.set_printoptions(precision=4, suppress=True)

def calculate_old_pose():
    left_arm_joint = np.array([1.57, 2.75, -2.3, 0.00, 0.00, 0.00])
    right_arm_joint = np.array([-1.57, 2.75, -2.3, 0.00, 0.00, 0.00])

    left_arm_tf_helper = utlis.TFHelper("torso_link4", "left_arm_link6", "mobiman", "urdf/R1_PRO/urdf/R1_PRO__URDF_20240904.urdf")
    left_transform = left_arm_tf_helper.calculate_transform(left_arm_joint)
    # print("left_transform: ", np.array(left_transform))
    # [ 0.0003  0.236  -0.4755  0.9748 -0.0001  0.0003  0.2231]

    right_arm_tf_helper = utlis.TFHelper("torso_link4", "right_arm_link6", "mobiman", "urdf/R1_PRO/urdf/R1_PRO__URDF_20240904.urdf")
    right_transform = right_arm_tf_helper.calculate_transform(right_arm_joint)
    # print("right transform: ", np.array(right_transform))
    return left_transform, right_transform
    # [ 0.0003 -0.236  -0.4755  0.9748 -0.0037 -0.0005 -0.2231]

def calculate_new_pose():
    left_arm_joint = np.array([1.57, 2.75, -2.3, 0.00, 0.00, 0.00])
    right_arm_joint = np.array([-1.57, 2.75, -2.3, 0.00, 0.00, 0.00])

    left_arm_tf_helper = utlis.TFHelper("torso_link4", "left_arm_link6", "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
    left_transform = left_arm_tf_helper.calculate_transform(left_arm_joint)
    # print("left_transform: ", np.array(left_transform))
    # [ 0.0003  0.236  -0.4504 -0.1577  0.689  -0.1578  0.6896]

    right_arm_tf_helper = utlis.TFHelper("torso_link4", "right_arm_link6", "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
    right_transform = right_arm_tf_helper.calculate_transform(right_arm_joint)
    # print("right transform: ", np.array(right_transform))
    # [ 0.0003 -0.236  -0.4504  0.1577  0.689   0.1578  0.6896]
    return left_transform, right_transform

def get_old_init_transform():
    left_pose = [0.0003, 0.236, -0.4755,  0.9748, -0.0001,  0.0003,  0.2231]
    right_pose = [0.0003, -0.236, -0.4755,  0.9748, -0.0037, -0.0005, -0.2231]
    left_transform = utlis.array2transform(left_pose[:3], left_pose[3:])
    right_transform = utlis.array2transform(right_pose[:3], right_pose[3:])
    return left_transform, right_transform

def get_new_init_transform():
    # left_pose = [0.0003,  0.236,  -0.4504, -0.1577,  0.689,  -0.1578,  0.6896]
    # right_pose = [0.0003, -0.236,  -0.4504,  0.1577,  0.689,   0.1578,  0.6896]
    left_pose, right_pose = calculate_new_pose()
    left_transform = utlis.array2transform(left_pose[:3], left_pose[3:])
    right_transform = utlis.array2transform(right_pose[:3], right_pose[3:])
    return left_transform, right_transform


class HDF5Updater():

    def __init__(self):
        self.left_new_tf_helper = utlis.TFHelper("torso_link4", "left_arm_link6", "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
        self.right_new_tf_helper = utlis.TFHelper("torso_link4", "right_arm_link6", "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
        self.floating_base_tf_helper = utlis.TFHelper("base_link", "torso_link4", "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
        self.old_init_left_mat, self.old_init_right_mat = get_old_init_transform()
        self.new_init_left_mat, self.new_init_right_mat = get_new_init_transform()
        self.left_delta_mat = np.dot(np.linalg.inv(self.old_init_left_mat), self.new_init_left_mat)
        self.right_delta_mat = np.dot(np.linalg.inv(self.old_init_right_mat), self.new_init_right_mat)
    
    def process_hdf5(self, input_file_path):
        h5_file = h5py.File(input_file_path, 'r+')
        self.update_observation(h5_file)
        self.update_action(h5_file, "left")
        self.update_action(h5_file, "right")
        h5_file.close()
    
    def process_dir(self, folder_path):
        h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
        for i in tqdm.tqdm(range(len(h5_files))):
            self.process_hdf5(h5_files[i])

    def update_observation(self, h5_file: h5py.File):
        left_arm_joints = h5_file["upper_body_observations/left_arm_joint_position"]
        new_left_ee_pose = []
        for arm_joint in left_arm_joints:
            new_ee_pose = self.left_new_tf_helper.calculate_transform(arm_joint)
            new_left_ee_pose.append(new_ee_pose)
        h5_file["upper_body_observations/left_arm_ee_pose"][...] = np.array(new_left_ee_pose)

        right_arm_joints = h5_file["upper_body_observations/right_arm_joint_position"]
        new_right_ee_pose = []
        for arm_joint in right_arm_joints:
            new_ee_pose = self.right_new_tf_helper.calculate_transform(arm_joint)
            new_right_ee_pose.append(new_ee_pose)
        h5_file["upper_body_observations/right_arm_ee_pose"][...] = np.array(new_right_ee_pose)

        torso_joints = h5_file["lower_body_observations/torso_joint_position"]
        new_floating_base_pose = []
        for torso_joint in torso_joints:
            new_fbase_pose = self.floating_base_tf_helper.calculate_transform(torso_joint)
            new_floating_base_pose.append(new_fbase_pose)
        h5_file["lower_body_observations/floating_base_pose"][...] = np.array(new_floating_base_pose)
    
    def update_action(self, h5_file: h5py.File, arm_name):
        field_name = f"upper_body_action_dict/{arm_name}_arm_ee_pose_cmd"
        delta_mat = self.left_delta_mat if arm_name == "left" else self.right_delta_mat
        old_action_poses = h5_file[field_name]
        new_action_poses = []
        for old_action_pose in old_action_poses:
            old_action_mat = utlis.array2transform(old_action_pose[:3], old_action_pose[3:])
            new_action_mat = np.dot(old_action_mat, delta_mat)
            new_action_pose = utlis.transform2array(new_action_mat)
            new_action_poses.append(new_action_pose)
        
        h5_file[field_name][...] = np.array(new_action_poses)


if __name__ == "__main__":
    updater = HDF5Updater()
    h5_file_path = sys.argv[1]
    updater.process_dir(h5_file_path)

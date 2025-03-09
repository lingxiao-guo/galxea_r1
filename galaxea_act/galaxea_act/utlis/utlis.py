import os
import re
import torch
import numpy as np
from galaxea_act.config.params import ArmType
from scipy.spatial.transform import Rotation as R
import random    
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def find_latest_checkpoint(folder):
    # 定义正则表达式来匹配文件名中的 epoch 数字
    checkpoint_files = [f for f in os.listdir(folder) if f.endswith('.ckpt')]
    if not checkpoint_files:
        return None, 0
    
    # 提取文件名中的 epoch 数字并找到最大的 epoch
    epoch_pattern = re.compile(r'epoch_(\d+)')
    max_epoch = -1
    latest_checkpoint = None
    
    for filename in checkpoint_files:
        match = epoch_pattern.search(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                latest_checkpoint = filename
    
    return latest_checkpoint, max_epoch




def get_arm_config(arm_type: ArmType, args_dict=None):
    obs_prefix = "upper_body_observations/"
    with_torso = args_dict['with_torso'] if args_dict is not None else False
    with_chassis = args_dict['with_chassis'] if args_dict is not None else False
    if arm_type == ArmType.LEFT or arm_type == ArmType.RIGHT:
        if args_dict is not None:
            if args_dict["tf"] == "joint_angles":
                qpos_dim = 7
                action_dim = 7
            elif args_dict["tf"] == "9d":
                qpos_dim = 13
                action_dim = 13
        if arm_type == ArmType.LEFT:
            camera_names = [obs_prefix + 'rgb_head', obs_prefix + 'rgb_left_hand']
        elif arm_type == ArmType.RIGHT:
            camera_names = [obs_prefix + 'rgb_head', obs_prefix + 'rgb_right_hand']
    elif arm_type == ArmType.BIMANUL:
        camera_names = [obs_prefix + 'rgb_head'] #, obs_prefix + 'rgb_left_hand', obs_prefix + 'rgb_right_hand']
        if args_dict is not None:
            if args_dict["tf"] == "joint_angles":
                qpos_dim = 7 + 7 
                action_dim = 7 + 7
            elif args_dict["tf"] == "9d":
                qpos_dim = 13 + 13
                action_dim = 13 + 13
    else:
        raise RuntimeError(f"unknown arm type: {arm_type}")
    
    if with_torso:
        qpos_dim += args_dict['with_torso']
        action_dim += args_dict['with_torso']

    if with_chassis:
        # qpos_dim += 3
        action_dim += 3
    
    if args_dict is not None:
        args_dict["camera_names"] = camera_names
        args_dict["qpos_dim"] = qpos_dim
        args_dict["action_dim"] = action_dim

    return camera_names, qpos_dim, action_dim


def flat_to_rotmtx_array(x):
    """Maps a 9D input vector onto SO(3) via symmetric orthogonalization.
    x: should be a single 9-element array.
    Output is a 3x3 matrix in SO(3).
    """
    # Reshape the 9D vector into a 3x3 matrix
    m = x.reshape(3, 3).T
    # Perform Singular Value Decomposition (SVD)
    u, s, vt = np.linalg.svd(m)
    # Compute determinant
    det = np.linalg.det(np.matmul(u, vt))
    # Adjust vt based on the determinant
    vt[:, -1] *= det
    # Compute the result matrix r
    r = np.matmul(u, vt)
    
    return r


def flat_to_rotmtx_array_batch(batch):
    """Maps a batch of 9D input vectors onto SO(3) via symmetric orthogonalization.
    batch: should be a 2D array of shape (batch_size, 9).
    Output is a 3D array of shape (batch_size, 3, 3) where each inner 3x3 matrix is in SO(3).
    """
    # Reshape the batch to (batch_size, 3, 3) and transpose each matrix
    batch_matrices = batch.reshape(-1, 3, 3).transpose(0, 2, 1)
    # Perform SVD on the entire batch at once
    u, s, vt = np.linalg.svd(batch_matrices)
    # Compute determinants for each matrix in the batch
    det = np.linalg.det(np.matmul(u, vt))
    # Adjust vt based on the determinant
    vt[:, :, -1] *= det[:, np.newaxis]
    # Compute the resulting batch of rotation matrices
    rot_matrices = np.matmul(u, vt)

    return rot_matrices


def quat_to_rpy(quat):
    rotation = R.from_quat(quat)
    rpy =rotation.as_euler('xyz')
    return rpy


def quat_to_rotmtx_batch(quat_batch):
    # Ensure the input is a NumPy array with shape (N, 4) where N is the batch size
    quat_batch = np.array(quat_batch)
    if quat_batch.shape[-1] != 4:
        raise ValueError("Each quaternion must have exactly 4 components [x, y, z, w]")
    # Create a rotation object from the batch of quaternions
    rotation = R.from_quat(quat_batch)
    # Convert the rotation object to a batch of rotation matrices
    rot_matrix_batch = rotation.as_matrix()  # This returns a (N, 3, 3) array of rotation matrices
    return rot_matrix_batch

def rotmtx_to_9d_batch(batch):
    """Transposes a batch of matrices and then flattens them.
    Args:
        batch (numpy.ndarray): A 3D array of shape (batch_size, rows, cols).
    Returns:
        numpy.ndarray: A 2D array of shape (batch_size, rows * cols) with flattened matrices.
    """
    # Transpose the batch of matrices
    transposed_batch = batch.transpose(0, 2, 1)  # Shape becomes (batch_size, cols, rows)
    # Flatten each transposed matrix
    flattened_batch = transposed_batch.reshape(batch.shape[0], -1)  # Shape becomes (batch_size, rows * cols)
    flattened_batch = flattened_batch.astype(np.float32)
    return flattened_batch

def rotmtx_to_9d(matrix):
    """Transposes a matrix and then flattens it.
    Args:
        matrix (numpy.ndarray): A 2D array of shape (rows, cols).
    Returns:
        numpy.ndarray: A 1D array of length rows * cols, where the matrix has been flattened.
    """
    # Transpose the matrix
    transposed_matrix = matrix.T  # Transpose the matrix (cols, rows)
    # Flatten the transposed matrix
    flattened_matrix = transposed_matrix.reshape(-1)  # Flatten the matrix into 1D
    return flattened_matrix


def transform_to_9d_batch(transform_array, gripper_pos_array=None):
    """
        transform_array: [batch_size, 7], columns arranged in [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        gripper_pos_array: [batch_size, 1]
    """
    transform_input = np.array(transform_array, dtype=np.float32)
    gripper_pos_array = np.array(gripper_pos_array, dtype=np.float32)
    rotation_matrix = quat_to_rotmtx_batch(transform_input[:,3:7])
    rot_9d_vec = rotmtx_to_9d_batch(rotation_matrix)

    if gripper_pos_array is None:
        source_list = [transform_input[:,0:3], rot_9d_vec]
    else:
        source_list = [transform_input[:,0:3], rot_9d_vec, gripper_pos_array]
    output = np.concatenate(source_list, axis = -1)
    return output

def transform_to_9d(transform_array, gripper_pos_array=None):
    transform_input = np.array(transform_array, dtype=np.float32)
    gripper_pos_array = np.array(gripper_pos_array, dtype=np.float32)
    rotation_matrix = quat_to_rotmtx_batch(transform_input[3:7])
    rot_9d_vec = rotmtx_to_9d(rotation_matrix)

    if gripper_pos_array is None:
        source_list = [transform_input[0:3], rot_9d_vec]
    else:
        source_list = [transform_input[0:3], rot_9d_vec, gripper_pos_array]
    output = np.concatenate(source_list, axis = -1)
    return output

def random_overlay(x, dataset, image_overlay):
    """Randomly overlay an image from Places"""
    alpha = image_overlay
    img, label = next(iter(dataset))
    img = img.numpy()
    img = img[:x.shape[0]]

    return (1-alpha) * x + (alpha) * img

class SaltAndPepperNoise:
    def __init__(self, prob_range=(0.0, 0.1)):
        """
        Initialize the SaltAndPepperNoise transformation.
        Args:
            prob_range (tuple): Min and max probability for the noise, e.g., (0.0, 0.1).
        """
        self.prob_min, self.prob_max = prob_range

    def __call__(self, img):
        """
        Apply salt-and-pepper noise to an image.
        Args:
            img (Tensor): Image tensor with shape (C, H, W) and values in [0, 1].
        Returns:
            Tensor: Noisy image.
        """
        if not torch.is_tensor(img):
            raise TypeError("Input image must be a PyTorch Tensor.")
        # Sample noise probability from a uniform distribution
        prob = random.uniform(self.prob_min, self.prob_max)
        b, c, h, w = img.shape
        # Generate a random mask
        mask = torch.rand(b, c, h, w)
        # Apply "salt" noise (set pixel to 1)
        img[mask < (prob / 2)] = 1.0
        # Apply "pepper" noise (set pixel to 0)
        img[mask > 1 - (prob / 2)] = 0.0
        return img

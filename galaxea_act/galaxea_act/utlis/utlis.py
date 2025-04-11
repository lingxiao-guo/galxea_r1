import os
import re
import torch
import numpy as np
from galaxea_act.config.params import ArmType
from scipy.spatial.transform import Rotation as R
import random    

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import hdbscan
import matplotlib.pyplot as plt

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
        camera_names = [obs_prefix + 'rgb_head']#, obs_prefix + 'rgb_left_hand']
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

import cv2 
def put_text(img, text, is_waypoint=False, font_size=1, thickness=2, position="top"):
    img = img.copy()
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (10, img.shape[0] - 60)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        str(text),
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    if is_waypoint:
        img = cv2.putText(
            img,
            "*",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
    return img

import torch
import torch.nn.functional as F
from scipy.special import digamma
import numpy as np

def k_nn_distance(x, k):
    """
    计算每个样本到其 K 个最近邻的距离。
    
    Args:
    - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
    - k (int): 最近邻的数量
    
    Returns:
    - distances (torch.Tensor): 每个样本到其 K 个最近邻的距离，形状为 (batch_size, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    
    # 计算样本之间的距离
    x_flat = x.view(batch_size, num_samples, -1)
    distances = torch.cdist(x_flat, x_flat)  # (batch_size, num_samples, num_samples)
    
    # 计算每个样本的 K 最近邻的距离
    k_distances, _ = torch.topk(distances, k + 1, dim=-1, largest=False)
    k_distances = k_distances[:, :, 1:]  # 排除自身的距离
    
    return k_distances

def kozachenko_leonenko_entropy(x, k=5):
    """
    使用 Kozachenko-Leonenko 估计器估计连续随机变量的熵。
    
    Args:
    - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
    - k (int): 最近邻的数量
    
    Returns:
    - entropy (torch.Tensor): 估计得到的熵，形状为 (batch_size, 1)
    """
    batch_size, num_samples, dim = x.size()
    
    # 计算 K 最近邻距离
    k_distances = k_nn_distance(x, k)
    
    # 计算每个样本的平均距离
    avg_distances = k_distances.mean(dim=2)
    
    # 计算熵
    # 使用 Digamma 函数计算
    digamma_k = torch.tensor(digamma(k), dtype=torch.float32, device=x.device)
    digamma_n = torch.tensor(digamma(num_samples), dtype=torch.float32, device=x.device)
    
    entropy = digamma_n - digamma_k - dim * torch.log(avg_distances).mean(dim=1, keepdim=True)
    
    return entropy

def gaussian_kernel(x, bandwidth):
    """
    计算高斯核函数。

    Args:
    - x (torch.Tensor): 样本点，形状为 (batch_size, num_samples, dim)
    - bandwidth (float): 核函数的带宽（标准差）

    Returns:
    - kernel_values (torch.Tensor): 高斯核的计算值，形状为 (batch_size, num_samples, num_samples)
    """
    batch_size, num_samples, dim = x.size()
    
    # 扩展维度以便计算距离矩阵
    x_i = x.unsqueeze(2)  # (batch_size, num_samples, 1, dim)
    x_j = x.unsqueeze(1)  # (batch_size, 1, num_samples, dim)
    
    # 计算距离矩阵
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (batch_size, num_samples, num_samples)
    
    # 计算高斯核
    kernel_values = torch.exp(-distances / (2 * bandwidth ** 2))
    
    return kernel_values


class KDE():
    def __init__(self, kde_flag=True, marginal_flag=True):
        self.flag = kde_flag
        self.marginal_flag = marginal_flag
    
    def kde_entropy(self,x,k=1):
        """
        使用核密度估计计算样本的熵，并对批次进行并行计算。

        Args:
        - x (torch.Tensor): 样本张量，形状为 (batch_size, num_samples, dim)
        - bandwidth (float): 核函数的带宽（标准差）

        Returns:
        - entropy (torch.Tensor): 计算得到的熵，形状为 (batch_size, 1)
        """
        batch_size, num_samples, dim = x.size()
        if self.flag:
            bandwidth = self.estimate_bandwidth(x[0])
            self.flag = False
        bandwidth = 1 #  for insertion, 0.001 for transfer
        # 计算高斯核
        kernel_values = gaussian_kernel(x, bandwidth)  # (batch_size, num_samples, num_samples)
    
        # 计算密度
        density = kernel_values.sum(dim=2) / num_samples  # (batch_size, num_samples)
        
        # 找到每个batch中密度最大的样本索引
        max_indices = torch.argmax(density, dim=1)  # (batch_size,)

        # 提取对应的样本点
        batch_indices = torch.arange(batch_size)  # 生成batch索引 [0, 1, ..., batch_size-1]
        max_density_points = x[batch_indices, max_indices, :]  # (batch_size, dim)

        # 计算对数密度
        log_density = torch.log(density + 1e-8)  # 添加平滑项以避免 log(0)
        
        # 计算熵
        entropy = -log_density.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        return entropy.squeeze(), max_density_points.squeeze()

    def estimate_bandwidth(self,x, rule='scott'):
    
        num_samples, dim = x.size()
    
        std = x.std(dim=0).mean().item()  # 计算各维度的标准差的平均值
        if rule == 'silverman':
            bandwidth = 1.06 * std * num_samples**(-1/5)
        elif rule == 'scott':
            bandwidth = std * num_samples**(-1/(dim + 4))
        else:
            raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")
    
        return bandwidth


from sklearn.ensemble import IsolationForest
import numpy as np

def remove_outliers_isolation_forest(data, contamination=0.25): # for DP: contamination = 0.25; ACT: contamination = 0.1
    model = IsolationForest(contamination=contamination)
    predictions = model.fit_predict(data.reshape(-1, 1))
    
    data = data.copy()  # 避免修改原数据

    # 处理首点
    if predictions[0] == -1:  # 如果首点是离群点
        next_idx = 1
        while next_idx < len(data) and predictions[next_idx] == -1:
            next_idx += 1
        if next_idx < len(data):  # 找到最近的非离群点
            data[0] = data[next_idx]

    # 处理尾点
    if predictions[-1] == -1:  # 如果尾点是离群点
        prev_idx = len(data) - 2
        while prev_idx >= 0 and predictions[prev_idx] == -1:
            prev_idx -= 1
        if prev_idx >= 0:  # 找到最近的非离群点
            data[-1] = data[prev_idx]

    # 处理中间的离群点
    for i in range(1, len(data) - 1):
        if predictions[i] == -1:  # 如果是离群点
            # 找到前一个非离群点
            prev_idx = i - 1
            while prev_idx >= 0 and predictions[prev_idx] == -1:
                prev_idx -= 1  # 跳过离群点

            # 找到后一个非离群点
            next_idx = i + 1
            while next_idx < len(data) and predictions[next_idx] == -1:
                next_idx += 1  # 跳过离群点

            # 如果能找到前后非离群点，用它们的均值替换
            if prev_idx >= 0 and next_idx < len(data):
                data[i] = (data[prev_idx] + data[next_idx]) / 2
            # 如果只能找到前一个非离群点，用前一个非离群点替换
            elif prev_idx >= 0:
                data[i] = data[prev_idx]
            # 如果只能找到后一个非离群点，用后一个非离群点替换
            elif next_idx < len(data):
                data[i] = data[next_idx]
                
    return data


def hdbscan_with_custom_merge(entropy, dir, rollout_id, plot=True):
    """
    使用HDBSCAN进行初步聚类，并根据规则进一步合并：
    - 第二个特征值小于0的点合并为一个簇；
    - 其余点合并为另一个簇；
    - 离群点 (-1 标签) 不参与合并。

    参数:
    X (array-like): 输入的数据
    dir (str): 保存图像的目录路径
    rollout_id (int): 用于标记图像文件名
    plot (bool): 是否绘制聚类结果

    返回:
    labels (array): 合并后的簇标签
    """
    # 排除离群点
    entropy = np.array(entropy)
    entropy_norm = (entropy-np.mean(entropy))/np.std(entropy)
    # entropy_norm[entropy_norm>2] = 0 
    entropy_norm = remove_outliers_isolation_forest(entropy_norm)
    entropy_norm = (entropy_norm-np.mean(entropy_norm))/np.std(entropy_norm)
    indices = np.arange(len(entropy_norm))
    indices = (indices-np.mean(indices))/np.std(indices)
    X = np.stack((indices,entropy_norm),axis=-1)
    # 初始化 HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(X)
    # TODO: add max_samples constraint
    # 初步聚类的标签
    initial_labels = clusterer.labels_

    # 将前?个点标记为离群点
    # initial_labels[:50] = -1
    
    # 后处理步骤，确保每个簇最多包含25个样本
    def split_large_clusters(labels, data, max_size=15):
        unique_labels = np.unique(labels)
        new_label = max(labels) + 1  # 用于分配新的簇标签

        for label in unique_labels:
            if label == -1:  # 跳过离群点
                continue

            # 获取当前簇的索引
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > max_size:
                # 如果簇的大小超过max_size，进行拆分
                cluster_points = data[cluster_indices]
                
                # 基于某些策略进行簇的拆分（例如按照距离拆分）
                # 这里用的是简单的按顺序分割的方法，可以根据需要更改为其他策略
                num_splits = len(cluster_indices) // max_size + (len(cluster_indices) % max_size > 0)
                
                for i in range(num_splits):
                    split_indices = cluster_indices[i * max_size:(i + 1) * max_size]
                    labels[split_indices] = new_label
                    new_label += 1  # 为每个拆分簇分配一个新的标签

        return labels
    
    # 拆分过大的簇
    initial_labels = split_large_clusters(initial_labels, X)
    
    # 获取非离群点的唯一标签
    unique_labels = np.unique(initial_labels[initial_labels >= 0])  # 排除噪声点 (-1)
    
    # 初始化合并后的标签
    refined_labels = np.full_like(initial_labels, -1)  # 默认所有点为离群值

    # 合并规则：
    # - 第二个特征 < 0 的点分为一类（合并到 0 类）
    # - 其余点分为另一类（合并到 1 类）
    for label in unique_labels:
        # 获取当前簇的点
        cluster_points = X[initial_labels == label]
        
        # 判断当前簇的第二个特征的值是否全部小于 0
        if  np.all(cluster_points[:, 1] < 0): # 0
            refined_labels[initial_labels == label] = 0  # 合并到第 0 类
        else:
            refined_labels[initial_labels == label] = -1  # 合并到第 1 类

    # 可视化原图
    if plot:
        plt.figure(figsize=(10, 6))
        # entropy[entropy>(np.mean(entropy)+2*np.std(entropy))] = 0
        plt.plot(np.arange(len(entropy_norm)), entropy_norm, marker='o', markersize=5)  # 使用点标记每个数据点
        # plt.ylim([0, np.mean(entropy)+2*np.std(entropy)])
        plt.title('1D Data Plot')
        plt.xlabel('Timestep')
        plt.ylabel('Entropy')
        plt.grid(True)  # 添加网格线
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-entropy-curve.png"))
        plt.close()

    # 可视化初步聚类结果
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=initial_labels, cmap='viridis', marker='o')
        plt.title('HDBSCAN Initial Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-raw.png"))
        plt.close()

    # 可视化合并后的结果
    if plot:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=refined_labels, cmap='viridis', marker='o')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Refined Cluster Label', rotation=270, labelpad=15)
        plt.title('HDBSCAN + Custom Merge Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-refine.png"))
        plt.close()
    return np.abs(refined_labels)

def kmeans_clustering(data,dir, rollout_id, max_clusters=10, plot_results=True):
    """
    使用 KMeans 对一维或二维数据进行自动分段。

    参数：
    - data (array-like): 输入数据，可以是一维或二维数组。
    - max_clusters (int): 最大尝试的簇数量。
    - plot_results (bool): 是否绘制可视化图表。

    返回：
    - labels (ndarray): 每个数据点的聚类标签。
    - optimal_clusters (int): 自动确定的簇数量。
    """
    silhouette_scores = []
    models = []

    # Step 1: 尝试不同的簇数量
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
        models.append((kmeans, labels))

    # Step 2: 找到最佳簇数（Silhouette 分数最大）
    optimal_clusters = np.argmax(silhouette_scores) + 2  # 索引偏移+2 对应簇数
    best_model, best_labels = models[optimal_clusters - 2]

    if plot_results:
        # 绘制 Silhouette 分数随簇数变化
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o", label="Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis")
        plt.grid()
        plt.axvline(optimal_clusters, color="r", linestyle="--", label="Optimal k")
        plt.legend()
        plt.show()

        # 可视化聚类结果（仅支持二维数据）
        if data.shape[1] == 2:
            plt.figure(figsize=(8, 5))
            plt.scatter(data[:, 0], data[:, 1], c=best_labels, cmap="viridis", label="Clusters")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("KMeans Clustering Result")
            plt.grid()
            plt.legend()
            plt.savefig(
                os.path.join(dir, f"plot/rollout{rollout_id}-kmeans.png")
            )
    return best_labels, optimal_clusters

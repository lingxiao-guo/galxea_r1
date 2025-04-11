import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def process_group(group, valid_indices, new_group):
    """
    递归处理组中的数据集，根据有效索引筛选数据。
    :param group: 当前组
    :param valid_indices: 有效的时间步索引
    :param new_group: 新的组
    """
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            # 筛选数据集
            original_data = item[:]
            filtered_data = original_data[valid_indices]
            new_group.create_dataset(key, data=filtered_data)
        elif isinstance(item, h5py.Group):
            # 递归处理子组
            new_subgroup = new_group.create_group(key)
            process_group(item, valid_indices, new_subgroup)
            
def clean_h5_files(input_folder, output_folder="clean", threshold=0.001):
    # 确保输出文件夹存在
    output_folder = os.path.join(input_folder,'../',output_folder)
    plot_folder = os.path.join(input_folder,'../','plot/')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)
    # 获取所有.h5文件
    h5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]

    # 定义需要处理的四个key
    keys_to_process = [
        "upper_body_action_dict/left_arm_ee_pose_cmd",
        "upper_body_action_dict/left_arm_gripper_position_cmd",
        "upper_body_action_dict/right_arm_ee_pose_cmd",
        "upper_body_action_dict/right_arm_gripper_position_cmd"
    ]
    obs_keys = [
        "/upper_body_observations/left_arm_ee_pose",
        "/upper_body_observations/left_arm_gripper_position",
        "/upper_body_observations/right_arm_ee_pose",
        "/upper_body_observations/right_arm_gripper_position",
    ]

    # 遍历每个.h5文件
    for file_name in h5_files:
        print(f'Filter {file_name}...')
        input_path = os.path.join(input_folder, file_name)
        
        output_path = os.path.join(output_folder, file_name)
        plot_path = os.path.join(plot_folder, file_name)
        
        # 打开原始文件
        with h5py.File(input_path, 'r') as src_file:
            # 读取四个key的数据
            data_list = []
            for key in keys_to_process:
                data_list.append(src_file[key][:])
            
            # 沿最后一个维度拼接数据
            concat_data = np.concatenate(data_list, axis=-1)
            
            # 计算斜率并确定需要保留的索引
            keep_indices = [True] * len(concat_data)
            for i in range(1, len(concat_data)):
                slope = np.linalg.norm((concat_data[i] - concat_data[i-1]),axis=-1)
                if slope < threshold:
                    keep_indices[i] = False
            
            
            # 创建新的h5文件并保存清洗后的数据
            with h5py.File(output_path, 'w') as new_file:
            # 递归处理所有组和数据集
                process_group(src_file, keep_indices, new_file)
                obs_list = []
                for key in obs_keys:
                    obs_list.append(new_file[key][:])
                obs = np.concatenate(obs_list, axis=-1)
                        
            # 沿最后一个维度拼接数据
            concat_data = np.concatenate(data_list, axis=-1)            
            # 绘制清洗后的曲线图
            cleaned_concat_data = concat_data[keep_indices]
            n_dims = cleaned_concat_data.shape[1]
            n_cols = 4
            n_rows = int(np.ceil(n_dims / n_cols))  # 计算需要的行数

            # 创建一个画布，设置整体大小
            plt.figure(figsize=(20, 15))  # 调整画布大小以适应4x4布局

            # 遍历每个维度
            for i in range(n_dims):
                # 创建子图，位置为第 (i // n_cols) 行第 (i % n_cols) 列
                plt.subplot(n_rows, n_cols, i + 1)
                # 绘制当前维度的数据
                plt.plot(cleaned_concat_data[:, i], label=f'target {i+1}')
                plt.plot(obs[:, i], label=f'obs {i+1}')
                # 添加标题和标签
                plt.title(f'Dim {i+1}')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()

            # 调整子图间距
            plt.tight_layout()

            # 保存图像
            plt.savefig(os.path.join(plot_folder, f'{os.path.splitext(file_name)[0]}_curves.png'))
            # 关闭画布
            plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_h5.py <input_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    clean_h5_files(input_folder)
import os
from tqdm import tqdm
import h5py
import numpy as np
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from pyquaternion import Quaternion as pyQuaternion
import kdl_parser_py.urdf

class TFInfoHelper:

    def __init__(self, urdf_path, root_link="torso_link4", tip_link="left_arm_link6"):
        # 解析URDF并生成KDL树
        self.robot = URDF.from_xml_file(urdf_path)
        success, self.tree = kdl_parser_py.urdf.treeFromUrdfModel(self.robot)
        if not success:
            raise ValueError("Failed to parse URDF and create KDL tree")

        # 从KDL树中获取链条（root_link -> tip_link）
        self.chain = self.tree.getChain(root_link, tip_link)
        # 创建正向运动学求解器
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)
    
    def process_h5file(self, h5_file_path):
        self.add_tf_info(h5_file_path, "observations/qpos", "observations/tf")
        self.add_tf_info(h5_file_path, "action", "action_tf")

    def process_h5files_in_directory(self, folder_path):
        """对文件夹及其子文件夹下的所有h5文件执行process_h5file操作"""
        # 创建一个列表来保存所有h5文件的路径
        h5_files = []

        # 使用os.walk递归遍历所有子文件夹，寻找.h5文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".h5"):
                    h5_files.append(os.path.join(root, file))

        # 使用tqdm显示处理进度
        for h5_file in tqdm(h5_files, desc="Processing h5 files"):
            self.process_h5file(h5_file)


    def add_tf_info(self, h5_file_path, read_entry, write_entry):
        # 打开HDF5文件进行读取和写入
        with h5py.File(h5_file_path, 'r+') as f:
            # 提取指定条目数据
            qpos_data = f[read_entry][:]  # 获取数据

            # 检查数据的形状是否符合要求
            if qpos_data.shape[1] < 6:
                raise ValueError("qpos array does not have enough joint angles (minimum 6 required).")

            # 初始化一个列表来保存转换信息
            transform_data = []

            # 遍历每个时刻的关节角数据
            for joint_angles in qpos_data:  # 仅取前6个作为关节角
                # 创建一个KDL的JntArray，用于存储关节角度
                q = kdl.JntArray(6)
                for i in range(6):
                    q[i] = joint_angles[i]
                
                # 计算末端执行器的正向运动学
                end_effector_frame = kdl.Frame()
                self.fk_solver.JntToCart(q, end_effector_frame)

                # 提取位置信息 (pos_x, pos_y, pos_z)
                pos = np.array([end_effector_frame.p[0], end_effector_frame.p[1], end_effector_frame.p[2]])

                # 提取四元数信息 (quat_x, quat_y, quat_z, quat_w)
                rot = end_effector_frame.M
                rot_matrix = np.array([[rot[0,0], rot[0,1], rot[0,2]],
                                            [rot[1,0], rot[1,1], rot[1,2]],
                                            [rot[2,0], rot[2,1], rot[2,2]]])
                quat = pyQuaternion(matrix=rot_matrix)

                # 保存提取到的数据
                transform_data.append([*pos, quat.x, quat.y, quat.z, quat.w, joint_angles[6]])

            # 将数据转换为numpy数组
            transform_data = np.array(transform_data)

            # 检查write_entry是否已存在，存在则删除以更新数据
            if write_entry in f:
                del f[write_entry]

            # 保存转换后的数据到指定条目
            f.create_dataset(write_entry, data=transform_data)  # 保存数据到指定HDF5条目

if __name__ == "__main__":
    import sys
    urdf_path = "/home/user/r1-ws/src/r1_urdf/r1_urdf_V104/URDF_R1_V1_0_4/urdf/URDF_R1_V1_0_4.urdf"
    root_link = "torso_link4"
    tip_link = "left_arm_link6"
    h5_file_path = sys.argv[1]
    tf_info_helper = TFInfoHelper(urdf_path, root_link, tip_link)
    # tf_info_helper.process_h5files_in_directory(h5_file_path)
    tf_info_helper.process_h5file(h5_file_path)
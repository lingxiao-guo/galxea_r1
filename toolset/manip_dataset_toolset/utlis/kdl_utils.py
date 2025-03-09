import os
import rospkg
import numpy as np
import PyKDL as kdl
import kdl_parser_py.urdf

from urdf_parser_py.urdf import URDF
from pyquaternion import Quaternion as pyQuaternion


def array2transform(position, quat_xyzw):
    mat = np.zeros((4, 4))
    mat[0, 3] = position[0]
    mat[1, 3] = position[1]
    mat[2, 3] = position[2]
    mat[3, 3] = 1.0

    quat = pyQuaternion(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    rot_mat = quat.rotation_matrix
    mat[:3, :3] = rot_mat

    return mat

def transform2array(transform_mat: np.array):
    rot_mat = transform_mat[:3, :3]
    quat = pyQuaternion(matrix=rot_mat)
    return np.array([transform_mat[0, 3], transform_mat[1, 3], transform_mat[2, 3], quat.x, quat.y, quat.z, quat.w])


class TFHelper:

    def __init__(self, root_link="torso_link4", tip_link="left_arm_link6", urdf_package_name="URDF_R1_V1_0_4", rel_path="urdf/URDF_R1_V1_0_4_fixed_torso.urdf"):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(urdf_package_name)
        urdf_path = os.path.join(package_path, rel_path)

        # 解析URDF并生成KDL树
        self.robot = URDF.from_xml_file(urdf_path)
        success, self.tree = kdl_parser_py.urdf.treeFromUrdfModel(self.robot)
        if not success:
            raise ValueError("Failed to parse URDF and create KDL tree") 
        # 从KDL树中获取链条（root_link -> tip_link）
        self.chain = self.tree.getChain(root_link, tip_link)
        # 创建正向运动学求解器
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

        # 从KDL链条中获取自由度数量
        self.dof = self.chain.getNrOfJoints()
        print(f"Number of degrees of freedom from {root_link} to {tip_link}: {self.dof}")

    def calculate_transform(self, joint_angles):
        # 检查数据的形状是否符合要求
        if len(joint_angles) < self.dof:
            raise ValueError(f"qpos array does not have enough joint angles (minimum {self.dof} required).")
        # 创建一个KDL的JntArray，用于存储关节角度
        q = kdl.JntArray(self.dof)
        for i in range(self.dof):
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
        quat2 = pyQuaternion(matrix=rot_matrix)
        # 保存提取到的数据
        result2 = [*pos, quat2.x, quat2.y, quat2.z, quat2.w]
        return result2

if __name__ == "__main__":
    root_link = "base_link"
    tip_link = "torso_link4"
    tf_info_helper = TFHelper(root_link, tip_link, "r1_v2_0_0", "urdf/r1_v2_0_0.urdf")
    qpos_data = np.array([1.57, 0.2, 1.79, 0])
    q2 = tf_info_helper.calculate_transform(qpos_data)
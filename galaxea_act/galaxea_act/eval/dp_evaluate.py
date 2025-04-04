import os
import time
import rospy
import torch
# import click
import imageio
import numpy as np
from threading import Thread
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import pickle
import argparse
from einops import rearrange
import matplotlib.pyplot as plt
import galaxea_act.utlis.utlis as utlis
from galaxea_act.utlis.utlis import get_arm_config
from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.algos.act_policy import ACTPolicy
from galaxea_act.algos.diffusion_policy import DiffusionPolicy
from galaxea_act.config.constants import IMAGE_WIDTH, IMAGE_HEIGHT
from galaxea_act.eval.infer_ros_node import InferRosNode
from galaxea_act.utlis.utlis import flat_to_rotmtx_array
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Transform,PoseStamped
from galaxea_act.config.constants import TEXT_EMBEDDINGS, TASK_INDEX_MAP
DEBUG_FLAG = False

########################## Constants for multi-process ###########################
T = 600
T_PRE = 30
T_POST = 300
N_OBS = 1
N_ACTIONS = 8
pre_actions = None # todo: set to default qpos

def get_task_emb(filename: str, use_one_hot=False):
    selected_task_index = 22
    for key, value in TASK_INDEX_MAP.items():
        if key in filename:
            selected_task_index = value
            break
    if use_one_hot:
        return selected_task_index
    else:
        return TEXT_EMBEDDINGS[selected_task_index-22]


def create_policy(args_dict):
    # command line parameters
    ckpt_dir = args_dict['ckpt_dir']
    args_dict['use_one_hot_task'] = False  # todo(dongke) 暂时默认不开启multi-task

    ckpt_name = 'policy_epoch_99_seed_0.ckpt'  # todo(dongke) hardcode for now
    # minmax_long: can't close the gripper completely
    # minmax mask: sometimes can't close the gripper, sometimes can't open the gripper
    # schedule mask: random, sometimes can close the gripper, but sometimes can't

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = DiffusionPolicy(args_dict)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    return policy

def saveimg_helper(input_image, output_path):
    image = (input_image * 255).to('cpu').numpy().astype(np.uint8)
    image = rearrange(image, 'c h w -> h w c')
    print("image shape: ", image.shape)
    imageio.imwrite(output_path, image)

# Using multi-process to mitigate model inference delay
class DPEvaluator(object):

    def __init__(self, args_dict) -> None:
        self.args_dict = args_dict
        self.with_torso = args_dict["with_torso"]
        self.with_chassis = args_dict["with_chassis"]
        self.wrc_demo = args_dict["wrc_demo"]
        np.random.seed(args_dict["seed"])        
        self.tick_times = 0

        self.arm_type = ArmType(args_dict['arm_type'])
        self.camera_names, self.qpos_dim, self.action_dim = get_arm_config(self.arm_type, args_dict)
        self.tf_type = args_dict['tf']
        self.ros_node = InferRosNode(self.arm_type, IMAGE_WIDTH, IMAGE_HEIGHT, self.with_torso, self.with_chassis, self.tf_type)
        policy = create_policy(args_dict)
        self.policy = policy 

        ckpt_dir = args_dict['ckpt_dir']
        self.temporal_agg = args_dict['temporal_agg']

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')  
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']  # todo(dongke) image的处理是写在model里的
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        self.qpos_post_process = lambda s_qpos: s_qpos * stats['qpos_std'] + stats['qpos_mean']

        # load environment
        self.query_frequency = args_dict['chunk_size']
        self.chunk_size = args_dict['chunk_size']
        self.latest_action_cache = None  # 用于存放policy最新推理出来的动作，常用于非temporal aggregation
        self.all_time_actions = torch.zeros([self.chunk_size, self.chunk_size, self.action_dim]).cuda() 
        if self.temporal_agg:
            self.query_frequency = 4
       
        self.qpos_store = []
        self.target_qpos_store = []

        # multiprocess shared memory
        shm_qpos_states = SharedMemory(name="qpos_states")
        shm_head_rgbs = SharedMemory(name="head rgbs")
        shm_actions = SharedMemory(name="actions")

        self.qpos_states = np.ndarray((T, 26), dtype=np.float32, buffer=shm_qpos_states.buf)
        self.actions = np.ndarray((T, 26), dtype=np.float32, buffer=shm_actions.buf)
        self.head_rgbs = np.ndarray((T, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32, buffer=shm_head_rgbs.buf)
           
    def start(self):
        self.ros_node.start(self.args_dict["start_flag"])
    
    def tick(self):
        """ 主要是推理获取动作
        """
        curr_time = get_curr_time()
        raw_action = self.actions[curr_time]
        action = self.post_process(raw_action)
            
        with torch.inference_mode():
            if True: 
                obs_dict = self.ros_node.get_observation()
                if obs_dict is None:
                    print(f"no valid observation at tick times {self.tick_times}")
                    return
                
                qpos, curr_image = self._process_obs_helper(obs_dict, self.tf_type)
                if DEBUG_FLAG:
                    print(f"received qpos {qpos.shape}")
                    print(f"received image {curr_image.shape}")
                    saveimg_helper(curr_image[0, 0], "head.png")
                    saveimg_helper(curr_image[0, 1], "hand.png")

                next_time = curr_time + 1
                with lock:
                    self.qpos_states[next_time] = qpos
                    self.head_rgbs[next_time] = curr_image
                    
                increment_curr_time(lock)
                     
        if DEBUG_FLAG:
            print(f"tick times {self.tick_times}, output action {action}")
        if self.arm_type == ArmType.BIMANUL:
            if self.tf_type == 'joint_angles':
                self.ros_node.pub_action(ArmType.LEFT, gripper_pos=action[6], joint_command=action[:6])
                self.ros_node.pub_action(ArmType.RIGHT, gripper_pos=action[13], joint_command=action[7:13])
            elif self.tf_type == '9d':
                action_left = action[0:13]
                action_right = action[13:26]
                transform_msg_left, gripper_pos_left = self.act_to_pose(action_left)
                transform_msg_right, gripper_pos_right = self.act_to_pose(action_right)
                self.ros_node.pub_action(ArmType.LEFT, gripper_pos = gripper_pos_left, ee_transform = transform_msg_left)       
                self.ros_node.pub_action(ArmType.RIGHT, gripper_pos = gripper_pos_right, ee_transform = transform_msg_right) 
                if self.with_torso: 
                    action_torso = action[26:26+self.with_torso]
                    self.ros_node.pub_torso(action_torso)     
                    if self.with_chassis:
                        action_chassis = action[26+self.with_torso:26+self.with_torso+3]
                        self.ros_node.pub_chassis(action_chassis) 
                if self.with_chassis and (not self.with_torso):
                    action_chassis = action[26:26+3]
                    self.ros_node.pub_chassis(action_chassis)  
        else:
            if self.tf_type == 'joint_angles':
                self.ros_node.pub_action(self.arm_type, gripper_pos=action[-1], joint_command=action[:6])
            elif self.tf_type == '9d':
                transform_msg, gripper_pos = self.act_to_pose(action[0:13])
                self.ros_node.pub_action(self.arm_type, gripper_pos = gripper_pos, ee_transform = transform_msg)
                if self.with_torso:
                    action_torso = action[13:13+self.with_torso]
                    self.ros_node.pub_torso(action_torso)
                    if self.with_chassis:
                        action_chassis = action[13+self.with_torso:13+self.with_torso+3]
                        self.ros_node.pub_chassis(action_chassis) 
                if self.with_chassis and (not self.with_torso):
                    action_chassis = action[13:13+3]
                    self.ros_node.pub_chassis(action_chassis)  
        self.tick_times += 1

    def plot_qpos(self,qpos, target_qpos, filename, work_dir='plot'):    
        qpos = np.array(qpos)
        target_qpos = np.array(target_qpos)
        num_dims = qpos.shape[1]  # 获取维度数量
        timestep = qpos.shape[0]  # 获取时间步数
        # print(self.action_data.shape)
        cols =2  # 每行 4 个子图
        rows = (num_dims + cols - 1) // cols  # 自动计算行数

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4)) 
        axes = axes.flatten()  # 将 2D 子图数组展平成 1D
        
        for i in range(num_dims):
            axes[i].plot(range(timestep), qpos[:, i], label=f"real qpos {i+1}")
            axes[i].plot(range(timestep), target_qpos[:, i], label=f"target qpos {i+1}")
            axes[i].set_title(f"Dimension {i+1}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Value")
            axes[i].legend()
        
        # 隐藏多余的子图（如果子图数量大于16）
        for j in range(num_dims, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()  # 自动调整子图间距
        
        fig.savefig(filename)
        plt.close(fig)


    def _process_obs_helper(self, obs_dict,tf_type):
        if tf_type == 'joint_angles':
            qpos_numpy = obs_dict["qpos"]
        elif tf_type == '9d':
            arm_tf = obs_dict["arm_tf"]
            if self.arm_type == ArmType.LEFT or self.arm_type == ArmType.RIGHT:
                qpos_numpy = utlis.transform_to_9d(arm_tf[:7], arm_tf[7:8])
            else:
                left_qpos_numpy = utlis.transform_to_9d(arm_tf[:7], arm_tf[7:8])
                right_qpos_numpy = utlis.transform_to_9d(arm_tf[8:15], arm_tf[15:16])
                qpos_numpy = np.concatenate([left_qpos_numpy, right_qpos_numpy], axis=-1)
        if self.with_torso:
            torso_numpy = obs_dict["torso_feedback"]
            waist_numpy = torso_numpy[4-self.with_torso:4]
            qpos_numpy = np.concatenate((qpos_numpy,waist_numpy),axis=-1)    
        # if self.with_chassis:
        #     chassis_numpy = obs_dict["chassis_feedback"]
        #     qpos_numpy = np.concatenate((qpos_numpy,chassis_numpy),axis=-1)    
        qpos = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        curr_images = []
        for cam_name in self.camera_names:
            curr_image = rearrange(obs_dict[cam_name], 'h w c -> c h w')
            curr_images.append(curr_image)

        curr_image = np.stack(curr_images, axis=0) 
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

        return qpos, curr_image
    
    def run(self):
        """
        主要调用函数
        """

        if self.wrc_demo:
            print("move robot to its initial position now")
        else:
            input("Press Enter to start robot and moves it to the initial position")
        self.start()
        time.sleep(0.2)

        if self.wrc_demo:
            print("start inference now after 2 seconds")
            time.sleep(2)
        else:
            input("Press Enter to start inference")
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            self.tick()
            rate.sleep()
        
        print("receive termination signal")
        self.ros_node.end()
    
    def act_to_tf(self, action):
        """
        the bridge from the action generated by the ACT model and the final tf command that will be published to the arm
        """
        rot_mat_flat = action[3:12]
        rot_matrix_recovered = flat_to_rotmtx_array(rot_mat_flat)
        rotation_recovered = R.from_matrix(rot_matrix_recovered)
        quat_recovered = rotation_recovered.as_quat()
        # Create a Transform message
        transform_msg = Transform()
        # Set translation (x, y, z)
        transform_msg.translation.x = action[0]
        transform_msg.translation.y = action[1]
        transform_msg.translation.z = action[2]
        # Set rotation (as a quaternion: x, y, z, w)
        transform_msg.rotation.x = quat_recovered[0]
        transform_msg.rotation.y = quat_recovered[1]
        transform_msg.rotation.z = quat_recovered[2]
        transform_msg.rotation.w = quat_recovered[3]
        gripper_pos = action[12]
        return transform_msg, gripper_pos

    def act_to_pose(self, action):
        """
        the bridge from the action generated by the ACT model and the final tf command that will be published to the arm
        """
        rot_mat_flat = action[3:12]
        rot_matrix_recovered = flat_to_rotmtx_array(rot_mat_flat)
        rotation_recovered = R.from_matrix(rot_matrix_recovered)
        quat_recovered = rotation_recovered.as_quat()
        # Create a Transform message
        transform_msg = PoseStamped()# Transform()
        # Set translation (x, y, z)
        transform_msg.pose.position.x = action[0]
        transform_msg.pose.position.y = action[1]
        transform_msg.pose.position.z = action[2]
        # Set rotation (as a quaternion: x, y, z, w)
        transform_msg.pose.orientation.x = quat_recovered[0]
        transform_msg.pose.orientation.y = quat_recovered[1]
        transform_msg.pose.orientation.z = quat_recovered[2]
        transform_msg.pose.orientation.w = quat_recovered[3]
        gripper_pos = action[12]
        return transform_msg, gripper_pos    


def create_shared_actions():
    shm_actions = SharedMemory(name="actions", create=True, size=T * 26 * np.float32().itemsize)
    actions = np.ndarray((T, 26), dtype=np.float32, buffer=shm_actions.buf)
    actions[:(T_PRE + N_ACTIONS)] = pad_after(pre_actions)
    return shm_actions

def create_shared_states():
    shm_qpos_states = SharedMemory(name="qpos_states", create=True, size=T * 26 * np.float32().itemsize)
    shm_head_rgbs = SharedMemory(name="head rgbs", create=True, size=T * IMAGE_WIDTH * IMAGE_HEIGHT * 3 * np.float32().itemsize)
    return shm_qpos_states, shm_head_rgbs

def create_shared_time():
    shm_curr_time = SharedMemory(name="curr_time", create=True, size=np.int64().itemsize)
    curr_time = np.ndarray((1,), dtype=np.int64, buffer=shm_curr_time.buf)
    curr_time[0] = 0
    return shm_curr_time

def get_curr_time():
    shm_curr_time = SharedMemory(name="curr_time")
    curr_time = np.ndarray((1,), dtype=np.int64, buffer=shm_curr_time.buf)
    return curr_time[0]

def increment_curr_time(lock):
    shm_curr_time = SharedMemory(name="curr_time")
    curr_time = np.ndarray((1,), dtype=np.int64, buffer=shm_curr_time.buf)
    with lock:
        curr_time[0] += 1

def clear_shm(name):
    shm = SharedMemory(name)
    shm.close()
    shm.unlink()

def pad_after(action_array, pad_length=N_ACTIONS):
    """
    param:
    action_array: (T, 26).
    """
    # print(action_array.shape)
    pad_array = np.zeros((action_array.shape[0] + pad_length, action_array.shape[1]))
    pad_array[:action_array.shape[0], :] = action_array
    last_action = action_array[-1, :]
    pad_array[action_array.shape[0]:, :] = last_action
    return pad_array
    
def infer_process(lock, stop_event, policy, device="cuda:0"):
    """
    """
    shm_actions = SharedMemory(name="actions")
    shm_qpos_states = SharedMemory(name="qpos_states")
    shm_head_rgbs = SharedMemory(name="head_rgb")
    actions = np.ndarray((T, 26), dtype=np.float32, buffer=shm_actions.buf)
    qpos_states = np.ndarray((T, 26), dtype=np.float32, buffer=shm_qpos_states.buf)
    head_rgbs = np.ndarray((IMAGE_WIDTH,IMAGE_HEIGHT, 3), dtype=np.float32, buffer=shm_head_rgbs.buf)

    while not stop_event.is_set():
        curr_time = get_curr_time()
        if T_PRE - N_ACTIONS < curr_time:
            print("infer at time: ", curr_time)
            qpos = qpos_states[(curr_time-N_OBS+1):(curr_time+1)]    # N_OBS x 7
            curr_image = head_rgbs[(curr_time-N_OBS+1):(curr_time+1)]
            # 形状注意做一下对齐
            data = curr_image, qpos, None, None, None

            time_infer_start = time.time()
            with torch.inference_mode():
                pred_actions = policy(data)
            time_infer_end = time.time()
            print(f"Time for inference: {time_infer_end - time_infer_start}")
            # print("prediction_dict['action'].shape: ", prediction_dict['action'].shape)

            pred_actions = pad_after(pred_actions).cpu().squeeze(0).numpy()
            
            with lock:
                actions[curr_time:(curr_time+2*N_ACTIONS)] = pred_actions
            if curr_time < T_PRE:
                with lock:
                    actions[curr_time:T_PRE] = pre_actions[curr_time:T_PRE]
                    
    # clear up                
    shm_actions.close()
    shm_qpos_states.close()
    shm_head_rgbs.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_parser()
    
    mp.set_start_method('spawn', force=True)
    # handle memory leakage
    shm_names = ["curr_time", "actions", "qpos_states",  "head_rgbs"]
    for shm_name in shm_names:
        try:
            clear_shm(shm_name)
            print(f"{shm_name} cleared")
        except:
            # print(f"{shm_name} does not exist")
            pass
    
    stop_event = mp.Event()
    lock = mp.Lock()
    shm_curr_time = create_shared_time()
    shm_actions = create_shared_actions()
    shm_qpos_states, shm_head_rgbs = create_shared_states()
    
    dp_evaluator = DPEvaluator(vars(parser.parse_args()))
    infer_proc = mp.Process(target=infer_process, args=(lock, stop_event, dp_evaluator.policy))
    # publish_proc = mp.Process(target=publish_process, args=(lock, stop_event, ros_node))

    infer_proc.start()
    # publish_proc.start()
    
    dp_evaluator.run()

    stop_event.set()
    infer_proc.join()
    # clear up
    shm_names = ["curr_time", "left_actions", "right_actions", "left_states", "right_states", "pcds"]
    for shm_name in shm_names:
        try:
            clear_shm(shm_name)
            print(f"{shm_name} cleared")
        except:
            print(f"{shm_name} does not exist")
            pass
    print("Shared memory cleared.")
        

    


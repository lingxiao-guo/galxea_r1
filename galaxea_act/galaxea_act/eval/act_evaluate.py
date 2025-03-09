import os
import time
import rospy
import torch
# import click
import imageio
import numpy as np
from threading import Thread

import pickle
import argparse
from einops import rearrange
import galaxea_act.utlis.utlis as utlis
from galaxea_act.utlis.utlis import get_arm_config
from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.algos.act_policy import ACTPolicy
from galaxea_act.config.constants import IMAGE_WIDTH, IMAGE_HEIGHT
from galaxea_act.eval.infer_ros_node import InferRosNode
from galaxea_act.utlis.utlis import flat_to_rotmtx_array
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Transform,PoseStamped
from galaxea_act.config.constants import TEXT_EMBEDDINGS, TASK_INDEX_MAP
DEBUG_FLAG = False

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

    ckpt_name = 'policy_best.ckpt'  # todo(dongke) hardcode for now

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(args_dict)
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

class ActEvaluator(object):

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

        # load environment
        self.query_frequency = args_dict['chunk_size']
        self.chunk_size = args_dict['chunk_size']
        self.latest_action_cache = None  # 用于存放policy最新推理出来的动作，常用于非temporal aggregation
        self.all_time_actions = torch.zeros([self.chunk_size, self.chunk_size, self.action_dim]).cuda() 
        if self.temporal_agg:
            self.query_frequency = 1
            
            
    def start(self):
        self.ros_node.start(self.args_dict["start_flag"])
    
    def tick(self):
        """ 主要是推理获取动作
        """
        rel_time_index = self.tick_times % self.chunk_size
        with torch.inference_mode():
            if self.tick_times % self.query_frequency == 0:
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

                if self.args_dict["multi_task"]:
                    task_emb = np.array(get_task_emb(self.args_dict.get("task_name", "")))
                    task_emb = torch.from_numpy(task_emb).float().cuda().unsqueeze(0)
                else:
                    task_emb=None
                self.latest_action_cache = self.policy(qpos, curr_image, task_emb=task_emb)
                
                # 更新对应时刻的action
                self.all_time_actions[rel_time_index] = self.latest_action_cache

            if self.latest_action_cache is None:
                print("warning, no available action")
                return

            if self.temporal_agg:
                past_num = min(self.chunk_size, self.tick_times + 1)
                # 往前数pat_num帧，每次往后偏移1个时刻，然后组合动作
                row_indexes = torch.arange(rel_time_index,  rel_time_index - past_num, -1)
                col_indexes = torch.arange(0, past_num, 1)
                actions_for_curr_step = self.all_time_actions[row_indexes, col_indexes]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = np.flip(exp_weights / exp_weights.sum()).copy()  # act原始论文里，越旧的动作权重越高，因此需要flip
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.latest_action_cache[:, rel_time_index]
            
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser = get_parser()
    act_evaluator = ActEvaluator(vars(parser.parse_args()))
    act_evaluator.run()



        

    


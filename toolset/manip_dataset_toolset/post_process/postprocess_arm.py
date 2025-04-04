#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从rosbag到hdf5文件的后处理脚本，本脚本专注处理非图像类数据，目前主要是上半身数据
"""
import rosbag
import numpy as np
from geometry_msgs.msg import Transform, Twist, PoseStamped
import manip_dataset_toolset.utlis.postprocess_utlis as utlis  # 保持数据处理脚本的相对独立，这样能单独分发出去
import time

GRIPPER_COMMAND_TOPIC = "/motion_control/position_control_gripper"
ARM_EE_COMMAND_TOPIC = "/motion_target/target_pose_arm"
ARM_JOINT_COMMAND_TOPIC = "/hdas/feedback_arm" #"/motion_target/target_joint_state_arm"
ARM_JOINT_STATES_TOPIC = "/hdas/feedback_arm"
ARM_GRIPPER_TOPIC = "/hdas/feedback_gripper"
ARM_POSE_TOPIC = "/relaxed_ik/motion_control/pose_ee_arm"
LEFT_ARM_SUFFIX = "_left"
RIGHT_ARM_SUFFIX = "_right"

class ArmPostProcessor(object):

    def __init__(self, arm_type: utlis.ArmType, task_space_cmd: bool, logger):
        self.logger = logger
        self.arm_type = arm_type
        self.task_space_cmd = task_space_cmd

    def process(self, messages, reference_timestamps, index_array):
        # Process each type of messages
        time_count = time.time()
        joint_state_dict_list = self.process_arm_joint_observation(messages, reference_timestamps, index_array)
        gripper_dict_list = self.process_arm_gripper_observation(messages, reference_timestamps, index_array)
        pose_state_dict_list = self.process_arm_pose_observation(messages, reference_timestamps, index_array)
        arm_cmd_dict_list = self.process_arm_command(messages, reference_timestamps, index_array)
        data_dict_list = []
        for i in range(len(joint_state_dict_list)):
            data_dict_i = {}
            data_dict_i.update(joint_state_dict_list[i])
            data_dict_i.update(gripper_dict_list[i])
            data_dict_i.update(arm_cmd_dict_list[i])
            data_dict_i.update(pose_state_dict_list[i])
            data_dict_list.append(data_dict_i)
        print(f"finish arm processing, time costs: {time.time()-time_count}")
        return data_dict_list

    def read_all_messages(self, input_bag):
        messages = {
            ARM_JOINT_STATES_TOPIC + LEFT_ARM_SUFFIX: [],
            ARM_JOINT_STATES_TOPIC + RIGHT_ARM_SUFFIX: [],
            ARM_GRIPPER_TOPIC + LEFT_ARM_SUFFIX: [],
            ARM_GRIPPER_TOPIC + RIGHT_ARM_SUFFIX: [],
            ARM_POSE_TOPIC + LEFT_ARM_SUFFIX: [],
            ARM_POSE_TOPIC + RIGHT_ARM_SUFFIX: [],
            ARM_EE_COMMAND_TOPIC + LEFT_ARM_SUFFIX: [],
            ARM_EE_COMMAND_TOPIC + RIGHT_ARM_SUFFIX: [],
            ARM_JOINT_COMMAND_TOPIC + LEFT_ARM_SUFFIX: [],
            ARM_JOINT_COMMAND_TOPIC + RIGHT_ARM_SUFFIX: [],
            GRIPPER_COMMAND_TOPIC + LEFT_ARM_SUFFIX: [],
            GRIPPER_COMMAND_TOPIC + RIGHT_ARM_SUFFIX: [],
        }
        for topic, msg, t in input_bag.read_messages():
            if topic in messages:
                messages[topic].append((msg, t))
        return messages

    def process_arm_joint_observation(self, messages, reference_timestamps, index_array):
        data_dict = {}
        for target_topic in [ARM_JOINT_STATES_TOPIC + LEFT_ARM_SUFFIX, ARM_JOINT_STATES_TOPIC + RIGHT_ARM_SUFFIX]:
            # hardcode for single arm
            # timestamps, joint_positions, joint_velocities = self.load_joint_state(messages[ARM_JOINT_STATES_TOPIC + LEFT_ARM_SUFFIX])
            timestamps, joint_positions, joint_velocities = self.load_joint_state(messages[target_topic])
            interpolated_joints = utlis.interpolate_1d(reference_timestamps, timestamps, joint_positions)
            interpolated_velocities = utlis.interpolate_1d(reference_timestamps, timestamps, joint_velocities)
            arm_prefix = "left" if "left" in target_topic else "right"
            data_dict[f"/upper_body_observations/{arm_prefix}_arm_joint_position"] = interpolated_joints[:, 0:6]
            data_dict[f"/upper_body_observations/{arm_prefix}_arm_joint_velocity"] = interpolated_velocities[:, 0:6]
        data_dict_list = utlis.dict_to_dict_list(data_dict, index_array)
        return data_dict_list

    def process_arm_gripper_observation(self, messages, reference_timestamps, index_array):
        data_dict = {}
        for target_topic in [ARM_GRIPPER_TOPIC + LEFT_ARM_SUFFIX, ARM_GRIPPER_TOPIC + RIGHT_ARM_SUFFIX]:
            # hardcode for single arm
            # timestamps, joint_positions, joint_velocities = self.load_joint_state(messages[ARM_GRIPPER_TOPIC + LEFT_ARM_SUFFIX])
            timestamps, joint_positions, joint_velocities = self.load_joint_state(messages[target_topic])
            interpolated_joints = utlis.interpolate_1d(reference_timestamps, timestamps, joint_positions)
            arm_prefix = "left" if "left" in target_topic else "right"
            data_dict[f"/upper_body_observations/{arm_prefix}_arm_gripper_position"] = interpolated_joints
        data_dict_list = utlis.dict_to_dict_list(data_dict, index_array)
        return data_dict_list

    @staticmethod
    def _arm_pose_h5_key_helper(end_effector_name, topic_name):
        if end_effector_name == "left_ee":
            return "/upper_body_observations/left_arm_ee_pose"
        if end_effector_name == "right_ee":
            return "/upper_body_observations/right_arm_ee_pose"
        if end_effector_name == "base_link":
            if "left" in topic_name:
                return "/upper_body_observations/left_arm_ee_pose"
            if "right" in topic_name:
                return "/upper_body_observations/right_arm_ee_pose"
            
        return ""

    def process_arm_pose_observation(self, messages, reference_timestamps, index_array):
        transform_dict = {}
        timestamp_dict = {}
        for target_topic in [ARM_POSE_TOPIC + LEFT_ARM_SUFFIX, ARM_POSE_TOPIC + RIGHT_ARM_SUFFIX]:
            # hardcode for single arm
            for msg, t in messages[target_topic]:
                msg_transform: PoseStamped = msg
                child_frame_id = msg.header.frame_id
                h5_key = self._arm_pose_h5_key_helper(child_frame_id, target_topic)
                if h5_key == "":
                    continue
                if h5_key not in transform_dict:
                    transform_dict[h5_key] = []
                    timestamp_dict[h5_key] = []
                transform_ref = msg_transform.pose
                pos = [transform_ref.position.x, transform_ref.position.y, transform_ref.position.z]
                quat = [transform_ref.orientation.x, transform_ref.orientation.y, transform_ref.orientation.z, transform_ref.orientation.w]
                transform_dict[h5_key].append(pos + quat)
                timestamp_dict[h5_key].append(t.to_sec())
        for key in transform_dict.keys():
            transform_dict[key] = np.array(transform_dict[key])
            timestamp_dict[key] = np.array(timestamp_dict[key])
            transform_dict[key] = utlis.interpolate_transform(reference_timestamps, timestamp_dict[key], transform_dict[key])
        
        # hardcode for single arm
        # transform_dict['/upper_body_observations/right_arm_ee_pose'] = transform_dict['/upper_body_observations/left_arm_ee_pose']
        
        data_dict_list = utlis.dict_to_dict_list(transform_dict, index_array)
        return data_dict_list

    def process_arm_command(self, messages, reference_timestamps, index_array):
        data_dict = {}
        if self.task_space_cmd:
            for target_topic in [ARM_EE_COMMAND_TOPIC + LEFT_ARM_SUFFIX, ARM_EE_COMMAND_TOPIC + RIGHT_ARM_SUFFIX]:
                # hardcode for single arm
                timestamps_ee, task_space_cmd_values = self.load_arm_ee_command(messages[target_topic])
                if len(task_space_cmd_values) > 0:
                    task_space_cmd_values = utlis.interpolate_transform(reference_timestamps, timestamps_ee, task_space_cmd_values)
                arm_prefix = "left" if "left" in target_topic else "right"
                data_dict[f"/upper_body_action_dict/{arm_prefix}_arm_ee_pose_cmd"] = task_space_cmd_values
                data_dict[f"/upper_body_action_dict/{arm_prefix}_arm_joint_position_cmd"] = np.array([])
        else:
            for target_topic in [ARM_JOINT_COMMAND_TOPIC + LEFT_ARM_SUFFIX, ARM_JOINT_COMMAND_TOPIC + RIGHT_ARM_SUFFIX]:
                # hardcode for single arm
                # timestamps_joint, joint_space_cmd_values,_ = self.load_joint_state(messages[ARM_JOINT_COMMAND_TOPIC + LEFT_ARM_SUFFIX])
                timestamps_joint, joint_space_cmd_values,_ = self.load_joint_command(messages[target_topic])
                if len(joint_space_cmd_values) > 0:
                    joint_space_cmd_values = utlis.interpolate_1d(reference_timestamps, timestamps_joint, joint_space_cmd_values)
                arm_prefix = "left" if "left" in target_topic else "right"
                data_dict[f"/upper_body_action_dict/{arm_prefix}_arm_ee_pose_cmd"] = np.array([])
                data_dict[f"/upper_body_action_dict/{arm_prefix}_arm_joint_position_cmd"] = joint_space_cmd_values[:, 0:6]
        for target_topic in [GRIPPER_COMMAND_TOPIC + LEFT_ARM_SUFFIX, GRIPPER_COMMAND_TOPIC + RIGHT_ARM_SUFFIX]:
            # hardcode for single arm
            # timestamps_gripper, gripper_cmd = self.load_gripper_command(messages[GRIPPER_COMMAND_TOPIC + LEFT_ARM_SUFFIX])
            timestamps_gripper, gripper_cmd = self.load_gripper_command(messages[target_topic])
            if len(gripper_cmd) > 0:
                gripper_cmd = utlis.interpolate_1d(reference_timestamps, timestamps_gripper, gripper_cmd)
            arm_prefix = "left" if "left" in target_topic else "right"
            data_dict[f"/upper_body_action_dict/{arm_prefix}_arm_gripper_position_cmd"] = gripper_cmd
        data_dict_list = utlis.dict_to_dict_list(data_dict, index_array)
        return data_dict_list

    def load_joint_state(self, messages):
        timestamps = []
        positions = []
        velocities = []
        for msg, t in messages:
            timestamps.append(t.to_sec())
            positions.append(list(msg.position))
            velocities.append(list(msg.velocity))
        utlis.frequency_helper(timestamps, "joint_state_topic", self.logger)
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        velocities = np.array(velocities)
        return timestamps, positions, velocities
    
    def load_joint_command(self, messages):
        timestamps = []
        positions = []
        velocities = []
        for msg, t in messages:
            timestamps.append(t.to_sec())
            positions.append(list(msg.p_des))
            velocities.append(list(msg.v_des))
        utlis.frequency_helper(timestamps, "joint_state_topic", self.logger)
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        velocities = np.array(velocities)
        return timestamps, positions, velocities

    def load_gripper_command(self, messages):
        timestamps = []
        positions = []
        for msg, t in messages:
            timestamps.append(t.to_sec())
            positions.append([msg.data])
        utlis.frequency_helper(timestamps, "gripper_command_topic", self.logger)
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        return timestamps, positions

    def load_arm_ee_command(self, messages):
        timestamps = []
        task_space_cmd_values = []
        for msg, t in messages:
            msg_pose: PoseStamped = msg
            msg_transform = msg_pose.pose
            pos = [msg_transform.position.x, msg_transform.position.y, msg_transform.position.z]
            quat = [msg_transform.orientation.x, msg_transform.orientation.y, msg_transform.orientation.z, msg_transform.orientation.w]
            task_space_cmd_values.append(pos + quat)
            timestamps.append(t.to_sec())
        utlis.frequency_helper(timestamps, "ee_command_topic", self.logger)
        timestamps = np.array(timestamps)
        task_space_cmd_values = np.array(task_space_cmd_values)
        return timestamps, task_space_cmd_values

    def load_arm_command(self, messages, target_topic):
        timestamps = []
        task_space_cmd_values = []
        joint_space_cmd_values = []
        gripper_cmd = []
        torso_cmd_values = []
        for msg, t in messages:
            timestamps.append(t.to_sec())
            if self.task_space_cmd:
                msg_transform: Transform = msg.transform
                pos = [msg_transform.translation.x, msg_transform.translation.y, msg_transform.translation.z]
                quat = [msg_transform.rotation.x, msg_transform.rotation.y, msg_transform.rotation.z, msg_transform.rotation.w]
                task_space_cmd_values.append(pos + quat)
            else:
                joint_space_cmd_values.append(msg.joint_command)
            if msg.torso_command is not None:
                msg_torso: Twist = msg.torso_command
                linear_vel = [msg_torso.linear.x, msg_torso.linear.y, msg_torso.linear.z]
                angular_vel = [msg_torso.angular.x, msg_torso.angular.y, msg_torso.angular.z]
                torso_cmd_values.append(linear_vel + angular_vel)
            gripper_cmd.append([msg.gripper_pos])
        utlis.frequency_helper(timestamps, target_topic, self.logger)
        timestamps = np.array(timestamps)
        task_space_cmd_values = np.array(task_space_cmd_values)
        joint_space_cmd_values = np.array(joint_space_cmd_values)
        gripper_cmd = np.array(gripper_cmd)
        torso_cmd_values = np.array(torso_cmd_values)
        if np.any(np.diff(timestamps) < 1e-5):
            print(f"warning topic {target_topic} has duplicate timestamps")
            self.logger.warning(f"warning topic {target_topic} has duplicate timestamps")
        return timestamps, task_space_cmd_values, joint_space_cmd_values, gripper_cmd, torso_cmd_values
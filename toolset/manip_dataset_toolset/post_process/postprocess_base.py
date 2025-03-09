#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从rosbag到hdf5文件的后处理脚本，本脚本专注处理非图像类数据，目前主要是下半身数据
"""
import rosbag
import numpy as np
import manip_dataset_toolset.utlis.postprocess_utlis as utlis
from geometry_msgs.msg import PoseStamped
import time

class BasePostProcessor(object):

    def __init__(self, logger):
        self.logger = logger

    def process(self, messages, reference_timestamps, index_array):
        time_count = time.time()
        # Step 1: Read all messages from the bag file once

        # Step 2: Separate messages by topics
        torso_observation_msgs = messages["/hdas/feedback_torso"]
        chassis_observation_msgs = messages["/hdas/feedback_chassis"]
        torso_tf_msgs = messages["/motion_control/pose_floating_base"]
        torso_action_msgs = messages["/motion_target/target_joint_state_torso"]
        chassis_action_msgs = messages["/motion_target/target_speed_chassis"]

        # Step 3: Process each topic's messages
        torso_data_dict = self.process_torso_observation(torso_observation_msgs, reference_timestamps, index_array)
        torso_tf_dict = self.process_torso_tf(torso_tf_msgs, reference_timestamps, index_array)
        torso_action_dict = self.process_torso_action(torso_action_msgs, reference_timestamps, index_array)
        chassis_data_dict = self.process_chassis_observation(chassis_observation_msgs, reference_timestamps, index_array)
        chassis_action_dict = self.process_chassis_action(chassis_action_msgs, reference_timestamps, index_array)
        data_dict_list = []
        for i in range(len(index_array)-1):
            data_dict = {}
            data_dict.update(torso_data_dict[i])
            data_dict.update(torso_tf_dict[i])
            data_dict.update(torso_action_dict[i])
            data_dict.update(chassis_data_dict[i])
            data_dict.update(chassis_action_dict[i])
            data_dict.update({
                "lower_body_action_dict/base_wheel_cmd": np.array([])
            })
            data_dict_list.append(data_dict)
        print(f"finish base processing, time costs: {time.time()-time_count}")
        return data_dict_list

    def read_all_messages(self, input_bag:rosbag.Bag):
        messages = {
            "/hdas/feedback_torso": [],
            "/hdas/feedback_chassis": [],
            "/motion_control/pose_floating_base": [],
            "/motion_target/target_joint_state_torso": [],
            "/motion_target/target_speed_chassis": []
        }
        for topic, msg, t in input_bag.read_messages():
            if topic in messages:
                messages[topic].append((msg, t))
        return messages

    def process_torso_observation(self, messages, reference_timestamps, index_array):
        data_dict = {}    
        
        timestamps_torso, torso_position, torso_velocity = self.load_joint_state(messages)
        if len(torso_position) > 0:
            torso_position = utlis.interpolate_1d(reference_timestamps, timestamps_torso, torso_position)
        data_dict["/lower_body_observations/torso_joint_position"] = torso_position
        data_dict_list = utlis.dict_to_dict_list(data_dict,index_array)
        return data_dict_list

    def process_chassis_observation(self, messages, reference_timestamps, index_array):
        data_dict = {}

        timestamps_torso, torso_position, torso_velocity = self.load_joint_state(messages)
        if len(torso_position) > 0:
            torso_position = utlis.interpolate_1d(reference_timestamps, timestamps_torso, torso_position)
        data_dict["/lower_body_observations/chassis_joint_position"] = torso_position
        data_dict_list = utlis.dict_to_dict_list(data_dict,index_array)
        return data_dict_list

    def process_torso_tf(self, messages, reference_timestamps, index_array):
        transform_list = []
        timestamp_list = []
        data_dict = {}

        for msg, t in messages:
            msg_transform: PoseStamped = msg
            transform_ref = msg_transform.pose
            pos = [transform_ref.position.x, transform_ref.position.y, transform_ref.position.z]
            quat = [transform_ref.orientation.x, transform_ref.orientation.y, transform_ref.orientation.z, transform_ref.orientation.w]
            transform_list.append(pos + quat)
            timestamp_list.append(t.to_sec())

        transform_list = np.array(transform_list)
        timestamp_list = np.array(timestamp_list) 
        transform_list = utlis.interpolate_transform(reference_timestamps, timestamp_list, transform_list) if len(transform_list) > 0 and len(timestamp_list) > 0 else np.array([]) 
        data_dict["/lower_body_observations/floating_base_pose"] = transform_list
        data_dict_list = utlis.dict_to_dict_list(data_dict, index_array)
        return data_dict_list
    
    def process_torso_action(self, messages, reference_timestamps, index_array):
        data_dict = {}  
        timestamps_torso, torso_cmd_values, joint_velocities = self.load_joint_state(messages)
        if len(torso_cmd_values) > 0:
            torso_cmd_values = utlis.interpolate_1d(reference_timestamps, timestamps_torso, torso_cmd_values) if len(torso_cmd_values) > 0 and len(timestamps_torso) > 0 else np.array([])
        data_dict[f"/lower_body_action_dict/torso_joint_position_cmd"] = torso_cmd_values  # torso_joint_position_cmd
        data_dict_list = utlis.dict_to_dict_list(data_dict,index_array)
        return data_dict_list

    def process_chassis_action(self, messages, reference_timestamps,index_array):
        data_dict = {}
        timestamps_torso, chassis_cmd_values = self.load_twist_state(messages)
        if len(chassis_cmd_values) > 0:
            chassis_cmd_values = utlis.interpolate_1d(reference_timestamps, timestamps_torso, chassis_cmd_values) if len(chassis_cmd_values) > 0 and len(timestamps_torso) > 0 else np.array([])
        data_dict[f"/lower_body_action_dict/chassis_target_speed_cmd"] = chassis_cmd_values  # torso_joint_position_cmd
        data_dict_list = utlis.dict_to_dict_list(data_dict,index_array)
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

        timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
        positions = np.array(positions)
        velocities = np.array(velocities)

        return timestamps, positions, velocities

    def load_twist_state(self, messages):
        timestamps = []
        velocities = []
        for msg, t in messages:
            timestamps.append(t.to_sec())
            velocities.append([msg.linear.x, msg.linear.y, msg.angular.z])
        utlis.frequency_helper(timestamps, "twist_state_topic", self.logger)

        timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
        velocities = np.array(velocities)

        return timestamps, velocities
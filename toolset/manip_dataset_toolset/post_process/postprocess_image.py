#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从rosbag到hdf5文件的后处理脚本，本脚本专注处理图像类数据
"""
import rosbag
import numpy as np
from cv_bridge import CvBridge
import manip_dataset_toolset.utlis.postprocess_utlis as utlis  # 保持数据处理脚本的相对独立，这样能单独分发出去
import time

ZED_HEAD_CAMERA_PREFIX = "/hdas/camera_head"
ZED_CAMERA_TOPIC = "/left_raw/image_raw_color/compressed"
ZED_CAMERA_DEPTH_TOPIC = "/depth/depth_registered"

LEFT_CAMERA_PREXIX = "/hdas/camera_wrist_left"
RIGHT_CAMERA_PREFIX = "/hdas/camera_wrist_right"
CAMERA_TOPIC = "/color/image_raw/compressed"
CAMERA_DEPTH_TOPIC = "/aligned_depth_to_color/image_raw"

class ImagePostProcessor(object):

    def __init__(self, arm_type:utlis.ArmType, logger):
        """
            input_bag: rosbag to process
            logger: logger from logging python module
        """
        self.arm_type = arm_type
        self.logger = logger
        self.cv_bridge = CvBridge()
    
    def process(self, messages, head_timestamps_list, index_array):
        time_count = time.time()
        img_data_dict = self.process_images(messages, head_timestamps_list, index_array)
        data_dict_list = [] 
        for i in range(len(img_data_dict)):
            data_dict_i = {}
            data_dict_i.update(img_data_dict[i])
            data_dict_list.append(data_dict_i)
        print(f"finish image processing, time costs: {time.time()-time_count}")
        return data_dict_list  # , head_timestamps

    def get_timestamp_list(self, input_bag):

        head_timestamps, timestamp_list = self._get_cutoff_timestamp(input_bag)
        index_array = np.array([0,len(head_timestamps)]) if len(timestamp_list) == 0 and head_timestamps.shape[0] > 0 else self._handle_cutoff(head_timestamps, timestamp_list) 
        self.logger.info(f"episode starts at {timestamp_list[0] - head_timestamps[0]} ends at {timestamp_list[-1] - head_timestamps[0]}") if len(timestamp_list) > 0 else self.logger.info("use all data with no breakpoint")

        return head_timestamps, index_array, timestamp_list  # , head_timestamps_all
    

    def process_images(self, image_messages, reference_timestamps, index_array):
        head_image_topic = ZED_HEAD_CAMERA_PREFIX + ZED_CAMERA_TOPIC
        # left_hand_topic = LEFT_CAMERA_PREXIX + CAMERA_TOPIC
        # right_hand_topic = RIGHT_CAMERA_PREFIX + CAMERA_TOPIC
        head_camera_depth_topic = ZED_HEAD_CAMERA_PREFIX + ZED_CAMERA_DEPTH_TOPIC
        # left_hand_depth_topic = LEFT_CAMERA_PREXIX + CAMERA_DEPTH_TOPIC
        # right_hand_depth_topic = RIGHT_CAMERA_PREFIX + CAMERA_DEPTH_TOPIC
        head_timestamps, head_images = self.load_rgb_images(image_messages[head_image_topic])
        data_dict = {"/upper_body_observations/rgb_head": head_images}
        # left_hand_timestamps, left_hand_images = self.load_rgb_images(image_messages[left_hand_topic])
        # aligned_timestamps_left_hand_images, left_hand_images = utlis.registrated_images(reference_timestamps, left_hand_timestamps, left_hand_images)
        # data_dict["/upper_body_observations/rgb_left_hand"] = left_hand_images
        # right_hand_timestamps, right_hand_images = self.load_rgb_images(image_messages[right_hand_topic])
        # aligned_timestamps_right_hand_images, right_hand_images = utlis.registrated_images(reference_timestamps, right_hand_timestamps, right_hand_images)
        # data_dict["/upper_body_observations/rgb_right_hand"] = right_hand_images
        head_depth_timestamps, head_depth_images = self.load_depth_images(image_messages[head_camera_depth_topic])
        _, head_depth_images = utlis.registrated_images(reference_timestamps, head_depth_timestamps, head_depth_images)
        data_dict["/upper_body_observations/depth_head"] = head_depth_images
        # left_hand_depth_timestamps, left_hand_depth_images = self.load_depth_images(image_messages[left_hand_depth_topic])
        # _, left_hand_depth_images = utlis.registrated_images(reference_timestamps, left_hand_depth_timestamps, left_hand_depth_images)
        # data_dict["/upper_body_observations/depth_left_hand"] = left_hand_depth_images
        # right_hand_depth_timestamps, right_hand_depth_images = self.load_depth_images(image_messages[right_hand_depth_topic])
        # _, right_hand_depth_images = utlis.registrated_images(reference_timestamps, right_hand_depth_timestamps, right_hand_depth_images)
        # data_dict["/upper_body_observations/depth_right_hand"] = right_hand_depth_images
        data_dict_list = utlis.dict_to_dict_list(data_dict, index_array)
        # save reference_timestamps, aligned_timestamps_left_hand_images, aligned_timestamps_right_hand_images into a npz file
        # np_ref = np.array(reference_timestamps)
        # np_lf = np.array(aligned_timestamps_left_hand_images)
        # np_rf = np.array(aligned_timestamps_right_hand_images)
        # np.savez("image_aligned_timestamps.npz", reference_timestamps=np_ref, left_hand_timestamps=np_lf, right_hand_timestamps=np_rf)
        return data_dict_list


    def load_rgb_images(self, messages):
        """
        Load depth images from a rosbag for multiple target topics and calculate the frequency of each topic.
        Args:
            input_bag (rosbag.Bag): The input rosbag to process.
            target_topics (list): List of target topics to process.
        Returns:
            dict: A dictionary where each key is a topic, and the value is a tuple containing:
                - A numpy array of image timesteps
                - A list of compressed depth images
        """
        topic={'timesteps': [], 'images': []} # topic_data = { for topic in target_topics}
        for msg, t in messages:  # for topic, msg, t in input_bag.read_messages(topics=target_topics):
            timestamp = t.to_sec()
            topic['timesteps'].append(timestamp)
            topic['images'].append(msg.data)
        timesteps = topic['timesteps']
        if timesteps:  # Only process if the topic has data
            utlis.frequency_helper(timesteps, "rgb_topics", self.logger)
        return np.array(topic['timesteps']), np.array(topic['images'])

    def load_depth_images(self, messages):
        """
        Load depth images from a rosbag for multiple target topics and calculate the frequency of each topic.
        Args:
            input_bag (rosbag.Bag): The input rosbag to process.
            target_topics (list): List of target topics to process.
        Returns:
            dict: A dictionary where each key is a topic, and the value is a tuple containing:
                - A numpy array of image timesteps
                - A list of compressed depth images
        """
        topic={'timesteps': [], 'images': []} 
        for msg, t in messages:  
            timestamp = t.to_sec()
            topic['timesteps'].append(timestamp)
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if image.dtype == np.float32:  # Handle depth images with float32 type
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                image[image < 0] = 0
                image = image * 1000.0
                image = np.clip(image, 0, 65535)
            image_compressed = utlis.compress_image_to_bytes(image.astype(np.uint16), extension='png')
            topic['images'].append(image_compressed)
        # for topic in target_topics:
        timesteps = topic['timesteps']
        if timesteps:  # Only process if the topic has data
            utlis.frequency_helper(timesteps, "depth_topic", self.logger)
        return np.array(topic['timesteps']), np.array(topic['images'])  # {topic: (np.array(data['timesteps']), data['images']) for topic, data in topic_data.items()}



    def _get_cutoff_timestamp(self, input_bag:rosbag.Bag):
        head_image_topic = ZED_HEAD_CAMERA_PREFIX + ZED_CAMERA_TOPIC
        target_topic = ["/breakpoint","/exception", head_image_topic]
        timestamps = []
        image_timesteps = []
        flag_exception = False
        for topic, msg, t in input_bag.read_messages(topics=target_topic):
            if topic == "/exception" and msg.data == True:
                flag_exception = True # break
            elif topic == "/breakpoint" and msg.data == True: 
                if flag_exception == False:
                    timestamps.append(t.to_sec())
            elif topic == head_image_topic:
                image_timesteps.append(t.to_sec())
        utlis.frequency_helper(image_timesteps, head_image_topic, self.logger)
        image_timesteps = np.array(image_timesteps)
        timestamps = np.array(timestamps)
        return image_timesteps, timestamps  # using the 奇偶性 of timestamps to tell if it contains odd number of trajs or even number of trajs # [0], timestamps[-1]

    def _handle_cutoff(self, reference_timestamp: np.ndarray, timestamps_list: np.ndarray):
        # Create a boolean mask where each element is True if the condition is met
        mask = reference_timestamp[:, np.newaxis] > timestamps_list 
        
        # Find the indices where the condition is first met for each element in timestamps_list
        index_array = np.argmax(mask, axis=0)
        
        # Generate a list of indices where the condition was met
        # index_array = index_array.tolist()
        
        return index_array # index_array

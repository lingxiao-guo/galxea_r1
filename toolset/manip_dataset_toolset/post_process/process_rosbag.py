#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from copy import deepcopy

import tqdm
import h5py
import rosbag
import logging
from multiprocessing import Pool, cpu_count


import traceback
import numpy as np
from datetime import datetime
import manip_dataset_toolset.utlis.postprocess_utlis as utlis
from manip_dataset_toolset.post_process.postprocess_arm import ArmPostProcessor
from manip_dataset_toolset.post_process.postprocess_image import ImagePostProcessor
from manip_dataset_toolset.post_process.postprocess_base import BasePostProcessor
from manip_dataset_toolset.utlis.zarr_replay_buffer import ReplayBuffer

ZED_HEAD_CAMERA_PREFIX = "/hdas/camera_head"
ZED_CAMERA_TOPIC = "/left_raw/image_raw_color/compressed"
ZED_CAMERA_DEPTH_TOPIC = "/depth/depth_registered"

LEFT_CAMERA_PREXIX = "/hdas/camera_wrist_left"
RIGHT_CAMERA_PREFIX = "/hdas/camera_wrist_right"
CAMERA_TOPIC = "/color/image_raw/compressed"
CAMERA_DEPTH_TOPIC = "/aligned_depth_to_color/image_raw"

GRIPPER_COMMAND_TOPIC = "/motion_control/position_control_gripper"
ARM_EE_COMMAND_TOPIC = "/motion_target/target_pose_arm"
ARM_JOINT_COMMAND_TOPIC = "/motion_control/control_arm" 
ARM_JOINT_STATES_TOPIC = "/hdas/feedback_arm"
ARM_GRIPPER_TOPIC = "/hdas/feedback_gripper"
ARM_POSE_TOPIC = "/relaxed_ik/motion_control/pose_ee_arm"
LEFT_ARM_SUFFIX = "_left"
RIGHT_ARM_SUFFIX = "_right"

class RosbagProcessor:

    def __init__(self, arm_type:utlis.ArmType, task_space_cmd:bool, log_file_dir:str, use_zarr:bool=False, num_parallel:int=40) -> None:
        """
        Args:
            arm_type(ArmType): 手臂类型
            task_space_cmd(bool): True means control command is given in the task space, False means joint space
        """
        self.arm_type = arm_type
        self.log_file_dir = log_file_dir
        self.task_space_cmd = task_space_cmd
        self.use_zarr = use_zarr
        self.replay_buffer = None
        self.init_log(log_file_dir)
        self.noise_topic_threshold = 0.2
        self.num_parallel = num_parallel

        self.arm_processor = ArmPostProcessor(arm_type, task_space_cmd, self.logger)
        self.image_processor = ImagePostProcessor(arm_type, self.logger)
        self.base_processor = BasePostProcessor(self.logger)

    def init_log(self, log_file_dir):
        if not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # 配置日志记录
        logging.basicConfig(
            level=logging.INFO, # 设置日志级别为INFO，低于此级别的日志将不会输出
            format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', # 设置日志格式
            datefmt='%Y-%m-%d %H:%M:%S', # 设置时间格式
            filename=os.path.join(log_file_dir, f'{current_time}.log'), # 设置日志文件名，如果不指定则默认输出到控制台
            filemode='w' # 以写模式打开日志文件，如果存在则覆盖
        )
        self.logger = logging.getLogger(__name__)



    def process_batch_file(self, args):
        """Helper function to process a single rosbag file in parallel."""
        bag_file, rosbag_dir, output_dir, task_id, debug_info, id_interval = args

        try:
            # Construct input and output file paths
            input_rosbag = bag_file # os.path.join(rosbag_dir, bag_file)

            # 获取相对路径并创建输出目录
            relative_path = os.path.relpath(bag_file, rosbag_dir)
            output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
            
            from pathlib import Path
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
           
            
            # 输出文件路径
            output_filename = os.path.splitext(os.path.basename(bag_file))[0] + '.h5'
            output_path = os.path.join(output_subdir, output_filename)
    
            # Call the method to process the rosbag
            self.process_rosbag(input_rosbag, output_path, task_id, debug_info, id_interval=id_interval)
            print("finish processing ", bag_file)

        except Exception as e:
            print(f"Unexpected exception caught while processing {bag_file}: {e}")
            print(traceback.format_exc())
            self.logger.error("Unexpected exception caught: %s", e)
            self.logger.error(traceback.format_exc())

    def process_dir(self, rosbag_dir: str, output_dir: str, task_id: str, debug_info=False, start_id=0, id_interval=1):
        if os.path.isfile(rosbag_dir):
            output_filename = os.path.splitext(os.path.basename(rosbag_dir))[0] + '.h5'
            output_path = os.path.join(output_dir, task_id, output_filename)
            try:
                self.process_rosbag(rosbag_dir, output_path, task_id, debug_info, id_interval=id_interval)
            except Exception as e:
                print(f"Unexpected exception caught: {e}")
                print(traceback.format_exc())
                self.logger.error("Unexpected exception caught: %s", e)
                self.logger.error(traceback.format_exc())
        elif os.path.isdir(rosbag_dir):
            # Get list of bag files
            bag_files = glob.glob(os.path.join(rosbag_dir, '**', '*.bag'), recursive=True)
            bag_files = sorted(bag_files)
            
            
            if not os.path.exists(output_dir):
                print(f"Creating output path {output_dir}")
                os.makedirs(output_dir)

            if self.use_zarr:
                self.replay_buffer = ReplayBuffer.create_from_path(output_dir, mode='w')


            MAX_PARALLEL = self.num_parallel
            # Prepare arguments for parallel processing
            args_list = []
            for bag_file in bag_files[start_id:]:
                args_list.append((bag_file, rosbag_dir, output_dir, task_id, debug_info, id_interval))
            print(f"start processing {len(args_list)} number of bags ....")
            print("......")
            print("......")
            
            # Split args_list into batches of size MAX_PARALLEL
            batch_size = MAX_PARALLEL
            batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]


            
            # if you want to stop the parallel processing, you can use the one line code below
            # self.process_batch_file(args_list[0])
            # Lambda function for processing each batch
            # process_batch = lambda pool, batch_args: list(tqdm.tqdm(pool.imap_unordered(self.process_batch_file, batch_args), total=len(batch_args)))
            def process_batch(pool, batch_args):
                for _ in tqdm.tqdm(pool.imap_unordered(self.process_batch_file, batch_args), total=len(batch_args)):
                    pass
            # Process each batch one at a time
            for batch in batches:
                with Pool(processes=MAX_PARALLEL) as pool:
                    process_batch(pool, batch)


            print(f"End of processing, produced {len(bag_files)} files in total")
        else:
            self.logger.error("Error: rosbag_dir is neither a file nor a directory")


    def process_rosbag(self, rosbag_path:str, output_hdf5_path=None, task_id=None, debug_info=False, id_interval=1):
        """ Extracts rosbag data and saves it to hdf5 file
        Args: 
            rosbag_path(str): rosbag path
            output_hdf5_path(str): output hdf5 file path, if set to None, this function will not save data

            Will use head camera's timestamp as the standard timestamp, and interpolate joint states / cmd values towards head camera's timestamps.
            Leave hand camera untouched, although there might be slight difference in time stamps btw hand and head cameras
        Returns:
            bool: True or False
        """
        if not os.path.exists(rosbag_path):
            print(rosbag_path)
            self.logger.error(f"rosbag {rosbag_path} not exists")
            return False
        
        print(f"begin process rosbag {rosbag_path}")
        self.logger.info(f"begin process rosbag {rosbag_path}")
        input_bag = rosbag.Bag(rosbag_path)
        head_timestamps, index_array, _ = self.image_processor.get_timestamp_list(input_bag)
        
        
        arm_messages = {
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
        base_messages = {
            "/hdas/feedback_torso": [],
            "/hdas/feedback_chassis": [],
            "/motion_control/pose_floating_base": [],
            "/motion_target/target_joint_state_torso": [],
            "/motion_target/target_speed_chassis": []
        }

        image_messages = {ZED_HEAD_CAMERA_PREFIX + ZED_CAMERA_TOPIC: [], 
                          LEFT_CAMERA_PREXIX + CAMERA_TOPIC: [], 
                          RIGHT_CAMERA_PREFIX + CAMERA_TOPIC: [], 
                          ZED_HEAD_CAMERA_PREFIX + ZED_CAMERA_DEPTH_TOPIC: [], 
                          LEFT_CAMERA_PREXIX + CAMERA_DEPTH_TOPIC: [], 
                          RIGHT_CAMERA_PREFIX + CAMERA_DEPTH_TOPIC: []}

        for topic, msg, t in input_bag.read_messages():
            
            if topic in arm_messages:
                arm_messages[topic].append((msg, t))
            elif topic in base_messages:
                base_messages[topic].append((msg, t))
            elif topic in image_messages:
                image_messages[topic].append((msg, t))
        
        input_bag.close()
        image_data_dict = self.image_processor.process(image_messages, head_timestamps, index_array)
        arm_data_dict = self.arm_processor.process(arm_messages, head_timestamps, index_array)
        base_data_dict = self.base_processor.process(base_messages, head_timestamps, index_array)
        
        for i in range(len(image_data_dict)):
            data_dict = {}
            data_dict.update(image_data_dict[i])
            data_dict.update(arm_data_dict[i])
            data_dict.update(base_data_dict[i])

            if self.use_zarr:
                empty_keys = []
                for key, value in data_dict.items():
                    if len(value) == 0:
                        empty_keys.append(key)
                for key in empty_keys:
                    del data_dict[key]
                self.replay_buffer.add_episode(data_dict, compressors='disk')
            else:
                head_msg_nums = len(data_dict['/upper_body_observations/rgb_head'])
                for key in list(data_dict.keys()):
                    if len(data_dict[key]) < self.noise_topic_threshold * head_msg_nums: #filter out the noisy topic whose number is less than 20% of the target topic
                        data_dict.pop(key)
                old_path = output_hdf5_path
                old_directory = os.path.dirname(old_path)
                old_file_name = os.path.basename(old_path)
                task_id_int = int(task_id)
                
                
                if (i % 2 == 0):# 第一段任务，如：抓起来,每个任务都分两段，所以这里除以2取余
                    task_id_str = f"{task_id_int:05d}"
                    subdirectory_path = os.path.join(old_directory, task_id_str)
                    if not os.path.exists(subdirectory_path):
                        os.makedirs(subdirectory_path, exist_ok=True)
                elif (i % 2 == 1):#第二段任务，如：放下去
                    task_id_int += id_interval
                    task_id_str = f"{task_id_int:05d}"
                    subdirectory_path = os.path.join(old_directory, task_id_str)
                    if not os.path.exists(subdirectory_path):
                        os.makedirs(subdirectory_path, exist_ok=True)
                print("subdirectory_path",subdirectory_path)
                old_file_name_dry = old_file_name[:-3]
                post_fix = '.h5'
                new_file_name_padded_zero = old_file_name_dry + f"-{int(i):03d}"# f"-{i}"
                new_file_name_format = new_file_name_padded_zero + post_fix
                new_path = os.path.join(subdirectory_path, new_file_name_format)
                self._save_h5_helper(data_dict, new_path, debug_info)
        return True
    
    def _save_h5_helper(self, data_dict:dict, output_h5_path, debug_info=False):
        if debug_info:
            for name, data in data_dict.items():
                print(f"field: {name} shape {data.shape}")
                if 'image' not in name:
                    print(f"max: {np.max(data, axis=0)}")
                    print(f"min: {np.min(data, axis=0)}")
                    print(f"mean: {np.mean(data, axis=0)}")
                else:
                    print(f"max: {np.max(data)}")
                    print(f"min: {np.min(data)}")
                    print(f"mean: {np.mean(data)}")

        # saves to hdf5 file
        if output_h5_path is not None:
            with h5py.File(output_h5_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = False
                for name, data in data_dict.items():
                    try:
                        # print(f"name:{name},data_shape:{len(data)}")
                        root[name] = data
                    except:
                        import pdb; pdb.set_trace()
                
                self.logger.info(f"save file to {output_h5_path}")

    
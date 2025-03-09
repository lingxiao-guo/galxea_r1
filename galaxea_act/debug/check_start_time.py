#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rosbag

def get_diff_time(arm_type, rosbag_path):
    input_bag = rosbag.Bag(rosbag_path)

    if arm_type == 1:
        target_topic = "/a1_robot_right/arm_command"
    else:
        target_topic = "/a1_robot_left/arm_command"
    for topic, msg, t in input_bag.read_messages(topics=target_topic):
        arm_start_time = t.to_sec()
        break

    head_camera_topic = "/zed2/zed_node/rgb_raw/image_raw_color/compressed"
    for topic, msg, t in input_bag.read_messages(topics=head_camera_topic):
        head_camera_start_time = t.to_sec()
        break
    diff_time = arm_start_time - head_camera_start_time
    input_bag.close()

    return diff_time


def process_dir(dir_path, arm_type):
    bag_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.bag')])
    for bag_file in bag_files:
        input_rosbag = os.path.join(dir_path, bag_file)
        diff_time = get_diff_time(arm_type, input_rosbag)
        if diff_time > 0.05:
            print(f"bag file: {bag_file} diff time {diff_time}")
        else:
            print("succ")
    

if __name__ == "__main__":
    import sys
    bag_dir = sys.argv[1]
    process_dir(bag_dir, 1)
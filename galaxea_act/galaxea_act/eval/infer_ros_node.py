import cv2
import time
import rospy
import numpy as np
from collections import deque
from cv_bridge import CvBridge
from functools import partial
from sensor_msgs.msg import CompressedImage, JointState
from galaxea_act.config.params import ArmType
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32

def pack_torso_cmd_message(torso_cmd,stack):
    cmd_msg = JointState()
    cmd_msg.position = list(stack[-1]["data"])
    cmd_msg.position[-1] = torso_cmd[-1]
    return cmd_msg

def pack_chassis_cmd_message(chassis_cmd,stack):
    cmd_msg = Twist()
    cmd_msg.linear.x = chassis_cmd[0]
    cmd_msg.linear.y = chassis_cmd[1]
    cmd_msg.angular.z = chassis_cmd[2]
    return cmd_msg

HEAD_CAMERA_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
LEFT_HAND_CAMERA_TOPIC = "/hdas/camera_wrist_left/color/image_raw/compressed"
RIGHT_HAND_CAMERA_TOPIC = "/hdas/camera_wrist_right/color/image_raw/compressed"
LEFT_JOINT_STATES_TOPIC = "/hdas/feedback_arm_left"
RIGHT_JOINT_STATES_TOPIC = "/hdas/feedback_arm_right"
LEFT_GRIPPER_TOPIC = "/hdas/feedback_gripper_left"
RIGHT_GRIPPER_TOPIC = "/hdas/feedback_gripper_right"

LEFT_ARM_EE_COMMAND_TOPIC = "/motion_target/target_pose_arm_left"
RIGHT_ARM_EE_COMMAND_TOPIC = "/motion_target/target_pose_arm_right"

LEFT_ARM_JOINT_COMMAND_TOPIC = "/motion_target/target_joint_state_arm_left"
RIGHT_ARM_JOINT_COMMAND_TOPIC = "/motion_target/target_joint_state_arm_right"

LEFT_GRIPPER_COMMAND_TOPIC = "/motion_control/position_control_gripper_left"
RIGHT_GRIPPER_COMMAND_TOPIC = "/motion_control/position_control_gripper_right"

"""负责组装observation, 发送action, 处理跟ROS相关的信息"""
class InferRosNode(object):

    def __init__(self, arm_type:ArmType, image_width, image_height, with_torso, with_chassis, tf_type) -> None:
        self.with_torso = with_torso
        self.with_chassis = with_chassis
        self.arm_type = arm_type
        self.image_w = image_width
        self.image_h = image_height
        self.msg_time_max_diff = 0.2

        self.head_camera_stack = deque(maxlen=10)
        self.left_hand_camera_stack = deque(maxlen=10)
        self.right_hand_camera_stack = deque(maxlen=10)
        self.left_joint_states_stack = deque(maxlen=10)
        self.right_joint_states_stack = deque(maxlen=10)
        self.left_tf_stack = deque(maxlen=10)
        self.right_tf_stack = deque(maxlen=10)
        self.left_gripper_stack = deque(maxlen=10)
        self.right_gripper_stack = deque(maxlen=10)
        if self.with_torso: self.torso_stack = deque(maxlen=10)
        if self.with_chassis: self.chassis_stack = deque(maxlen=10)
        self.obs_stack_dict = self._generate_obs_stack()

        self.sub_head_camera = None

        self.sub_left_hand_camera = None
        self.sub_right_hand_camera = None

        self.sub_left_joint_states = None
        self.sub_right_joint_states = None
        
        self.sub_left_tf = None
        self.sub_right_tf = None
        
        self.sub_left_gripper = None
        self.sub_right_gripper = None
        
        
        self.left_gripper_cmd_publisher = None
        self.left_arm_ee_command_publisher = None
        self.left_arm_joint_command_publisher = None
        
        self.right_gripper_cmd_publisher = None
        self.right_arm_ee_command_publisher = None
        self.right_arm_joint_command_publisher = None
        
        if self.with_torso: self.pub_torso_command = None
        
        self.tf_type = tf_type
        # image related
        self.br = CvBridge()
    
    def _generate_obs_stack(self):
        image_prefix = "upper_body_observations/"
        if self.arm_type == ArmType.LEFT:
            obs_dict = {
                image_prefix + "rgb_head": self.head_camera_stack,
                image_prefix + "rgb_left_hand": self.left_hand_camera_stack,
                "qpos": self.left_joint_states_stack,
                "arm_tf": self.left_tf_stack,
                "gripper": self.left_gripper_stack
            }
        elif self.arm_type == ArmType.RIGHT:
            obs_dict = {
                image_prefix + "rgb_head": self.head_camera_stack,
                image_prefix + "rgb_right_hand": self.right_hand_camera_stack,
                "qpos": self.right_joint_states_stack,
                "arm_tf": self.right_tf_stack,
                "gripper": self.right_gripper_stack
            }
        elif self.arm_type == ArmType.BIMANUL:
            obs_dict = {
                image_prefix + "rgb_head": self.head_camera_stack,
                image_prefix + "rgb_left_hand": self.left_hand_camera_stack,
                image_prefix + "rgb_right_hand": self.right_hand_camera_stack,
                "qpos_left": self.left_joint_states_stack,
                "qpos_right": self.right_joint_states_stack,
                "arm_tf_left": self.left_tf_stack,
                "arm_tf_right": self.right_tf_stack,
                "gripper_left": self.left_gripper_stack,
                "gripper_right": self.right_gripper_stack
            }
        else:
            raise RuntimeError(f"unknown arm type: {self.arm_type}")
        
        if self.with_torso: obs_dict["torso_feedback"] = self.torso_stack
        if self.with_chassis: obs_dict["chassis_feedback"] = self.chassis_stack
        return obs_dict

    def start(self, start_flag=True):
        rospy.init_node("act_inference_node") # Subscribe to images and joint states, then output arm_command

        self.sub_head_camera = rospy.Subscriber(HEAD_CAMERA_TOPIC, CompressedImage, 
                                                partial(self._camera_callback, image_stack=self.head_camera_stack))

        if self.arm_type in (ArmType.RIGHT, ArmType.BIMANUL):
            self.sub_right_hand_camera = rospy.Subscriber(RIGHT_HAND_CAMERA_TOPIC, CompressedImage, 
                                                         partial(self._camera_callback, image_stack = self.right_hand_camera_stack))
            self.sub_right_joint_states = rospy.Subscriber(RIGHT_JOINT_STATES_TOPIC, JointState, 
                                                          partial(self._joint_states_callback, joint_states_stack = self.right_joint_states_stack))
            self.sub_right_tf = rospy.Subscriber('/motion_control/pose_ee_arm_right', PoseStamped, partial(self._tf_callback, tf_stack = self.right_tf_stack, arm_type = ArmType.RIGHT))
            self.sub_right_gripper = rospy.Subscriber(RIGHT_GRIPPER_TOPIC, JointState, 
                                                          partial(self._gripper_callback, gripper_stack = self.right_gripper_stack))
            time.sleep(1)  # 创建完Subscriber和Publisher要等待一点时间，否则会丢包            

            if self.tf_type == '9d':
                self.right_arm_ee_command_publisher = rospy.Publisher(RIGHT_ARM_EE_COMMAND_TOPIC, PoseStamped, queue_size = 25)
            elif self.tf_type == 'joint_angles':
                self.right_arm_joint_command_publisher = rospy.Publisher(RIGHT_ARM_JOINT_COMMAND_TOPIC, JointState, queue_size = 25)
            self.right_gripper_cmd_publisher = rospy.Publisher(RIGHT_GRIPPER_COMMAND_TOPIC, Float32, queue_size = 25)
        if self.arm_type in (ArmType.LEFT, ArmType.BIMANUL):
            self.sub_left_hand_camera = rospy.Subscriber(LEFT_HAND_CAMERA_TOPIC, CompressedImage, 
                                                         partial(self._camera_callback, image_stack = self.left_hand_camera_stack))
            self.sub_left_joint_states = rospy.Subscriber(LEFT_JOINT_STATES_TOPIC, JointState, 
                                                          partial(self._joint_states_callback, joint_states_stack = self.left_joint_states_stack))
            self.sub_left_tf = rospy.Subscriber('/motion_control/pose_ee_arm_left', PoseStamped, partial(self._tf_callback, tf_stack = self.left_tf_stack, arm_type = ArmType.LEFT))
            self.sub_left_gripper = rospy.Subscriber(LEFT_GRIPPER_TOPIC, JointState, 
                                                          partial(self._gripper_callback, gripper_stack = self.left_gripper_stack))
            time.sleep(1)  # 创建完Subscriber和Publisher要等待一点时间，否则会丢包

            if self.tf_type == '9d':
                self.left_arm_ee_command_publisher = rospy.Publisher(LEFT_ARM_EE_COMMAND_TOPIC, PoseStamped, queue_size = 25)
            elif self.tf_type == 'joint_angles':
                self.left_arm_joint_command_publisher = rospy.Publisher(LEFT_ARM_JOINT_COMMAND_TOPIC, JointState, queue_size = 25)
            self.left_gripper_cmd_publisher = rospy.Publisher(LEFT_GRIPPER_COMMAND_TOPIC, Float32, queue_size = 25)
                
        if self.with_torso:
            self.sub_torso_feedback = rospy.Subscriber('/hdas/feedback_torso', JointState, partial(self._torso_callback, torso_stack = self.torso_stack))
            self.pub_torso_command = rospy.Publisher("/motion_target/target_joint_state_torso", JointState, queue_size = 25)
        if self.with_chassis:
            self.sub_chassis_feedback = rospy.Subscriber('/hdas/feedback_chassis', JointState, partial(self._chassis_callback, chassis_stack = self.chassis_stack))
            self.pub_chassis_command = rospy.Publisher("/motion_target/target_speed_chassis", Twist, queue_size = 25)
        print(f"start infer node with arm type {self.arm_type} and start message published")


    def end(self):
        for sub in [self.sub_head_camera, self.sub_left_hand_camera, self.sub_left_joint_states, self.sub_left_tf, 
                    self.sub_right_hand_camera, self.sub_right_joint_states, self.sub_right_tf]:
            if sub is not None:
                sub.unregister()
        print(f"end infer node")

    def clear_stack(self):
        for stack in self.obs_stack_dict.values():
            stack.clear()

    def get_observation(self):
        """获取观测数据，如果说某个观测没收到数据，或者不同观测之间的时间差距太大，则会返回None，否则返回一个字典

        Returns:
            dict
        """
        latest_msgs = {}
        min_time = time.time() + 1000
        max_time = -1
        for name, stack in self.obs_stack_dict.items(): # name=key, stack=value, since it is a dictionary
            if len(stack) == 0:
                print(f"obs {name} receives no message yet")  # todo(dongke) 有空的时候要改用logging模块
                return None
            else:
                latest_msg = stack[-1]
                msg_time = latest_msg["message_time"]
                min_time = min(min_time, msg_time)
                max_time = max(max_time, msg_time) 
                latest_msgs[name] = np.array(latest_msg["data"])
        
        if self.arm_type == ArmType.BIMANUL:
            left_gripper = latest_msgs["gripper_left"][-1]  # [-1] is also OK, since it is the only thing
            latest_msgs["arm_tf_left"] = np.append(latest_msgs["arm_tf_left"],left_gripper)
            latest_msgs["qpos_left"][-1] = left_gripper
            right_gripper = latest_msgs["gripper_right"][-1]
            latest_msgs["arm_tf_right"] = np.append(latest_msgs["arm_tf_right"],right_gripper)
            latest_msgs["qpos_right"][-1] = right_gripper
            latest_msgs["qpos"] = np.concatenate([latest_msgs["qpos_left"], latest_msgs["qpos_right"]], axis=-1)
            del latest_msgs["qpos_left"]
            del latest_msgs["qpos_right"]
            latest_msgs["arm_tf"] = np.concatenate([latest_msgs["arm_tf_left"], latest_msgs["arm_tf_right"]], axis=-1)
            del latest_msgs["arm_tf_left"]
            del latest_msgs["arm_tf_right"]
            del latest_msgs["gripper_left"]
            del latest_msgs["gripper_right"]
        else:
            gripper = latest_msgs["gripper"][-1]
            latest_msgs["arm_tf"] = np.append(latest_msgs["arm_tf"], gripper)
            latest_msgs["qpos"][-1] = gripper
            
        if max_time - min_time > self.msg_time_max_diff:
            print(f"warning max msg time difference: {max_time - min_time} is larger than {self.msg_time_max_diff}")
            for name, stack in self.obs_stack_dict.items():
                print("message name: ", name)
                print("time: ", stack[-1]["message_time"])
        
        return latest_msgs
    
    def pub_action(self, arm_type:ArmType, gripper_pos:float, joint_command=None, ee_transform=None):
        if joint_command is not None and len(joint_command) != 6:
            print(f"abnormal joint command size {len(joint_command)}")
        
        if (joint_command is None) and (ee_transform is None):
            rospy.logwarn("EXCEPT AT THE START AND AT THE END, OTHERWISE DON'T DO IT!")
        
        if arm_type == ArmType.LEFT:
            if self.tf_type == '9d':
                self.left_arm_ee_command_publisher.publish(ee_transform)
            elif self.tf_type == 'joint_angles':
                self.left_arm_joint_command_publisher.publish(joint_command)
            self.left_gripper_cmd_publisher.publish(gripper_pos)
        elif arm_type == ArmType.RIGHT:
            if self.tf_type == '9d':
                self.right_arm_ee_command_publisher.publish(ee_transform)
            elif self.tf_type == 'joint_angles':
                self.right_arm_joint_command_publisher.publish(joint_command)
            self.right_gripper_cmd_publisher.publish(gripper_pos)
        else:
            print("warning: unknown arm type: ", arm_type)

    def pub_torso(self, torso_cmd = None):
        if torso_cmd is not None and len(self.torso_stack)>=1:
            msg = pack_torso_cmd_message(torso_cmd, self.torso_stack)
            self.pub_torso_command.publish(msg)
        else:
            print("warning: no torso command!")

    def pub_chassis(self, chassis_cmd = None):
        if chassis_cmd is not None and len(self.chassis_stack)>=1:
            msg = pack_chassis_cmd_message(chassis_cmd, self.chassis_stack)
            self.pub_chassis_command.publish(msg)
        else:
            print("warning: no torso command!")
       
    def _camera_callback(self, msg:CompressedImage, image_stack=None):
        camera_time = msg.header.stamp.to_sec()
        img_camera = self.br.compressed_imgmsg_to_cv2(msg)
        img_camera = np.array(cv2.resize(img_camera, (self.image_w, self.image_h)))
        image_body_dict = {"message_time": camera_time, "data": img_camera}
        image_stack.append(image_body_dict)

    
    def _joint_states_callback(self, msg:JointState, joint_states_stack=None):
        joint_state_time = msg.header.stamp.to_sec()

        if len(msg.position) != 7:
            raise RuntimeError(f"abnormal joint state message, length should be 7: {msg.name}")
        data_dict = {
            "message_time": joint_state_time,
            "data": msg.position
        }
        joint_states_stack.append(data_dict)

    def _gripper_callback(self, msg:JointState, gripper_stack=None):
        gripper_time = msg.header.stamp.to_sec()

        if len(msg.position) != 1:
            raise RuntimeError(f"abnormal gripper state message, length should be 1: {msg.name}")
        data_dict = {
            "message_time": gripper_time,
            "data": msg.position
        }
        gripper_stack.append(data_dict)

    def _tf_callback(self, msg:PoseStamped, tf_stack=None, arm_type = None):  # not yet considering the bi-manual case
        if arm_type == ArmType.RIGHT:
            desired_child_frame_id = 'right_ee'
        elif arm_type == ArmType.LEFT:
            desired_child_frame_id = 'left_ee'
        transform = msg
        if (transform.header.frame_id == desired_child_frame_id):
            tf_time = transform.header.stamp.to_sec()
            # Convert translation and rotation to lists
            translation_list = [transform.pose.position.x, transform.pose.position.y, transform.pose.position.z]
            rotation_list = [transform.pose.orientation.x, transform.pose.orientation.y, transform.pose.orientation.z, transform.pose.orientation.w]
            # Combine the two lists
            combined_list = translation_list + rotation_list
            data_dict = {
                "message_time": tf_time,
                "data": combined_list
            }
            tf_stack.append(data_dict)

    def _torso_callback(self, msg:JointState, torso_stack = None):
        torso_time = msg.header.stamp.to_sec()

        if len(msg.position) != 4:
            raise RuntimeError(f"abnormal joint state message, length should be 7: {msg.name}")
        data_dict = {
            "message_time": torso_time,
            "data": msg.position
        }
        torso_stack.append(data_dict)

    def _chassis_callback(self, msg:JointState, chassis_stack = None):
        chassis_time = msg.header.stamp.to_sec()

        if len(msg.position) != 3:
            raise RuntimeError(f"abnormal joint state message, length should be 3: {msg.name}")
        data_dict = {
            "message_time": chassis_time,
            "data": msg.position
        }
        chassis_stack.append(data_dict)

if __name__ == "__main__":
    ros_node = InferRosNode(ArmType.BIMANUL, 320, 240)
    
    input("press enter to start listening")
    ros_node.start()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        obs_dict = ros_node.get_observation()
        for key, value in obs_dict.items():
            print(f"{key}: ", value.shape)
        rate.sleep()
    
    ros_node.end()



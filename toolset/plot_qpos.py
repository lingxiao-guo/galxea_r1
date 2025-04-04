import rosbag
import numpy as np
import matplotlib.pyplot as plt

# 读取bag文件
bag = rosbag.Bag('1-0000-20250319152818.bag', 'r')  # 请替换为实际路径

# 初始化存储容器
control_times = []
p_des_list = []
feedback_times = []
position_list = []

# 提取数据
for topic, msg, t in bag.read_messages(topics=['/motion_target/target_pose_arm_left', '/motion_control/pose_ee_arm_left']):
    current_time = t.to_sec()  # 使用ROS bag记录时间
    
    if topic == '/motion_target/target_pose_arm_left':
        # if len(msg.p_des) == 7:  # 确保数据维度正确
            control_times.append(current_time)
            # p_des_list.append(msg.p_des)
            x,y,z,rx,ry,rz,rw = msg.pose.position.x,msg.pose.position.y,msg.pose.position.z,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w
            qpos = np.stack((x,y,z,rx,ry,rz,rw),axis=0)
            p_des_list.append(qpos)
    
    elif topic == '/motion_control/pose_ee_arm_left':
        # if len(msg.position) == 7:  # 确保数据维度正确
            feedback_times.append(current_time)
            x,y,z,rx,ry,rz,rw = msg.pose.position.x,msg.pose.position.y,msg.pose.position.z,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w
            qpos = np.stack((x,y,z,rx,ry,rz,rw),axis=0)
            position_list.append(qpos)

# 转换为numpy数组并排序
control_times = np.array(control_times)
p_des_array = np.array(p_des_list)
feedback_times = np.array(feedback_times)
position_array = np.array(position_list)

# 按时间排序
control_sort_idx = np.argsort(control_times)
feedback_sort_idx = np.argsort(feedback_times)

control_times = control_times[control_sort_idx]
p_des_array = p_des_array[control_sort_idx]
feedback_times = feedback_times[feedback_sort_idx]
position_array = position_array[feedback_sort_idx]
D = position_array.shape[-1]
# 时间对齐（线性插值）
interpolated_positions = np.zeros((len(control_times), D))
for i in range(D):
    interpolated_positions[:,i] = np.interp(control_times, 
                                          feedback_times, 
                                          position_array[:,i])

# 转换为相对时间
t0 = control_times[0]
times = control_times - t0
p_des_array = p_des_array[:-1]
interpolated_positions = interpolated_positions[1:]
times = times[:-1]
print(p_des_array.shape)
print(interpolated_positions.shape)
# 绘制图形
plt.figure(figsize=(12, 18))
for i in range(D):
    plt.subplot(D, 1, i+1)
    plt.plot(times, p_des_array[:,i], label='Desired Position')
    plt.plot(times, interpolated_positions[:,i], label='Actual Position')
    plt.ylabel(f'Dimension {i+1}')
    plt.grid(True)
    plt.legend()

plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('qpos.jpg')

bag.close()
上传指令示例：
rsync -avh --progress /home/liusong/data_rosbagh5/bag/bag_raw/ {username}@192.168.0.249:/mnt/mnt_0/data/dataset/operation/raw/00039/
把rosbag数据的文件夹放在/data/${TASK_NAME}/raw 文件夹下
处理数据指令示例：
source /opt/ros/noetic/setup.bash
注：之前只在左臂上收集的数据，为了方便，把toolset/manip_dataset_toolset/post_process/postprocess_arm.py中改了三处self.load_joint_state和一处self.load_gripper_command，标记注释为hardcode, 记得改回来
python3 scripts/rosbag2hdf5.py ../data/${TASK_NAME}/raw/ ../data/${TASK_NAME}/h5 1   # 1是task id,应该随便给?
对EE pose control:
python3 scripts/rosbag2hdf5.py ../data/${TASK_NAME}/raw/ ../data/${TASK_NAME}/h5 1 --task_space_cmd
刷新四元数正负号示例(只适用于ee pose control)：
python3 scripts/h5_quaternion_regulation.py ../data/${TASK_NAME}/h5
自动2：8划分测试，训练集指令：
python3 scripts/split_training_set.py /data/operation/processed/00054
清理数据，filter掉静止帧：
对ee pose control
python scripts/filter_h5.py ../data/${TASK_NAME}/h5/00001

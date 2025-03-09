

# 安装
Run `bash install.sh` for creating conda environment and installing dependencies. 

# 训练

# 数据集格式
给定的`dataset_dir`下，应该还有两层文件夹，比如`pickpen/0710/*.h5`, 训练脚本会提取dataset_dir两层文件夹下的所有h5文件

## rosbag数据集处理
处理脚本： `galaxea_act/dataset/process_rosbag_dataset.py`

常见处理命令[对DP，chunk_size应该为16]
```
# joint angles
python galaxea_act/train.py --dataset_dir ../data/${TASK_NAME}/h5/clean  --ckpt_dir checkpoints/${POLICY_CLASS}/${TASK_NAME} --policy_class ACT --kl_weight 10 --chunk_size 15 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --seed 0 --temporal_agg --num_epochs 200 --lr 5e-5 --task_name ${TASK_NAME} --run_name ${TASK_NAME}  --arm_type 2 --tf joint_angles 
# ee pose
python galaxea_act/train.py --dataset_dir ../data/${TASK_NAME}/h5/clean  --ckpt_dir checkpoints/${POLICY_CLASS}/${TASK_NAME} --policy_class ACT --kl_weight 10 --chunk_size 15 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --seed 0 --temporal_agg --num_epochs 200 --lr 5e-5 --task_name ${TASK_NAME} --run_name ${TASK_NAME}  --arm_type 2 --tf 9d
```
对DP，建议可以把num_epochs搞到400
# 实机测试

命令示例：
```
python galaxea_act/eval/act_evaluate.py --ckpt_dir checkpoints/screwdriver --lr 5e-5 --kl_weight 10 --chunk_size 100 --hidden_dim 512 --dim_feedforward 3200 --seed 0 --temporal_agg --arm_type 0  --tf 9d
```
启动该程序以后，想要kill掉，可以ctrl + C, 然后执行`ps`，会有一个`pt_main_thread`的进程，用`kill -9 pid`, pid是该进程对应的pid号码

```
rosbag record /a1_robot_right/arm_command /a1_robot_right/joint_states /camera_head/color/image_raw/compressed /camera_right_hand/color/image_raw/compressed -O model_output_train.bag
```

相关topic名称：
```
/a1_robot_left/joint_states /a1_robot_left/arm_command /camera_left_hand/color/image_raw/compressed /a1_robot_right/joint_states /a1_robot_right/arm_command /camera_right_hand/color/image_raw/compressed  /zed2/zed_node/rgb_raw/image_raw_color/compressed /torso_feedback
```

replay播放topic:
```
/a1_robot_left/joint_states /camera_left_hand/color/image_raw/compressed /a1_robot_right/joint_states /camera_right_hand/color/image_raw/compressed  /zed2/zed_node/rgb_raw/image_raw_color/compressed
```


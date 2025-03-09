#!/bin/bash

# 自动训练脚本，使用者自己修改路径，但不要把修改过后的本脚本上传到git

python3 galaxea_act/dataset/process_rosbag_dataset.py /data/ManipulationDataset/Tape/HalfWay/ /data/ManipulationDatasetH5/Tape/LeftArm/Halfway tape_halfway
python3 galaxea_act/dataset/process_rosbag_dataset.py /data/ManipulationDataset/Tape/NearDeskDown/ /data/ManipulationDatasetH5/Tape/LeftArm/NearDeskDown/ tape_near_desk_down
python3 galaxea_act/dataset/process_rosbag_dataset.py /data/ManipulationDataset/Tape/NearDeskMiddle/ /data/ManipulationDatasetH5/Tape/LeftArm/NearDeskMiddle/ tape_near_desk_middle
python3 galaxea_act/dataset/process_rosbag_dataset.py /data/ManipulationDataset/Tape/NearDeskTilt/ /data/ManipulationDatasetH5/Tape/LeftArm/NearDeskTilt/ tape_near_desk_tilt
python3 galaxea_act/dataset/process_rosbag_dataset.py /data/ManipulationDataset/Tape/RandomPosition/ /data/ManipulationDatasetH5/Tape/LeftArm/RandomPosition/ tape_random_position

conda activate galaxea_act

python galaxea_act/train.py --dataset_dir /data/ManipulationDatasetH5/Tape/ --ckpt_dir checkpoints/tape0814 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --seed 0 --temporal_agg --num_epochs 10000 --lr 5e-5 --task_name tape0814 --run_name tape0814
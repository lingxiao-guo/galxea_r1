"""
PyTorch Infer of trained ACT model
minimal implementation
"""
import torch
import os
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from galaxea_act.algos.act_policy import ACTPolicy
from galaxea_act.dataset.episodic_dataset import EpisodicDatasetGalaxea
from galaxea_act.config.parser import get_parser
from galaxea_act.config.params import ArmType
from galaxea_act.dataset.episodic_dataset import generate_arm_feature_helper
from galaxea_act.utlis.utlis import load_model_from_checkpoint,get_arm_config

def count_h5_files(dataset_dir):
    # 查找目录下所有的.h5文件
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    # 返回文件数量
    return len(h5_files)


def torch_infer(args_dict):
    # Set additional configs
    arm_type = ArmType(args_dict['arm_type'])
    args_dict["camera_names"], qpos_dim, action_dim = get_arm_config(arm_type, args_dict)
    ckpt_dir = args_dict['ckpt_dir']
    ckpt_name = "policy_best.ckpt" # Default name
    # Load model
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(args_dict)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    # Change to evaluate mode
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    # Load dataset and get data
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl') # Load dataset statistics(mean,var)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    num_episodes = 0
    for directories in args_dict['dataset_dir']:
        num_episodes += count_h5_files(directories)
    dataset = EpisodicDatasetGalaxea(episode_ids=[_ for _ in range(num_episodes)],dataset_dir=args_dict['dataset_dir'],norm_stats=stats,num_episodes=num_episodes,
                                     chunk_size=args_dict['chunk_size'],camera_names=args_dict["camera_names"],tf_representation=args_dict["tf"],
                                     arm_type=arm_type,with_torso=args_dict['with_torso'],with_chassis=args_dict['with_chassis'],image_overlay=args_dict['image_overlay'])
    print("Loading dataset complete")
    # Get ready for multi trial scenarios, now only one trial len(dataset.trials) = 1
    for trial in dataset.trials:
        gt_action = generate_arm_feature_helper(trial, arm_type, True, args_dict["tf"] == "9d",
                                                with_torso=args_dict["with_torso"], with_chassis=args_dict["with_chassis"])
    inference_action = []
    num_steps = args_dict["num_steps"] if args_dict["num_steps"] is not None else gt_action.shape[0] # Partial replay
    start_time = args_dict["start_time"]
    range_x = np.arange(start_time,start_time+num_steps,1) # x-axis for gt
    fig, ax = plt.subplots(2,3,figsize=(15,10))
    for row in ax: # set plot range
        for axis in row:
            axis.set_ylim(-0.2, 0.2)
    ax[0][0].plot(range_x,gt_action[:num_steps,-3],label="x")
    ax[0][0].set_title("Chassis X")
    ax[0][1].plot(range_x,gt_action[:num_steps,-2],label="y")
    ax[0][1].set_title("Chassis Y")
    ax[0][2].plot(range_x,gt_action[:num_steps,-1],label="z")
    ax[0][2].set_title("Chassis Z")

    for i in range(start_time,start_time+num_steps,1):
        print(f"Start plotting the {i}-th timestep")
        image_data, qpos_data, action_data, is_pad, task_emb = dataset.__getitem__(0,start_ts=i)
        qpos_data = qpos_data.unsqueeze(0)
        image_data, qpos_data = image_data.cuda(), qpos_data.cuda()

        # Infer the model
        with torch.no_grad():
            action = policy(qpos_data,image_data) # Action shape (1,chunk_size, action_dim)
            action = action.squeeze(0).detach().cpu().numpy() # Reshape action to (chunk_size, action_dim) for convenience
            inference_action.append(action)
        x_axis = np.arange(i,min(i+args_dict['chunk_size'],start_time+num_steps),1)
        end_point = min(i+args_dict['chunk_size'],start_time+num_steps)-i # set plot range for chunks
        ax[1][0].plot(x_axis,action[:end_point,-3],label="x")
        ax[1][1].plot(x_axis,action[:end_point,-2],label="y")
        ax[1][2].plot(x_axis,action[:end_point,-1],label="z")
    plt.savefig(f"work_dirs/inference_visualize/inference_{start_time}_{start_time+num_steps}.png",dpi=300)
    # save fig to work_dirs/, increase dpi for better visualization
    print("Inference action length is:",len(inference_action))


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--start_time', type=int, required=False, default=0)
    parser.add_argument('--num_steps', type=int, required=False)
    torch_infer(vars(parser.parse_args()))

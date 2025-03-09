import os
import glob
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt
from einops import rearrange

from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.utlis.utlis import set_seed, get_arm_config
from galaxea_act.utlis.utlis import quat_to_rotmtx_batch, rotmtx_to_9d_batch, rotmtx_to_9d, flat_to_rotmtx_array_batch
from galaxea_act.algos.act_policy import ACTPolicy
import h5py


def visualize_differences_4(qpos_list,gt_list,infer_list,gt2_list=None, plot_path=None, ylim=None, label_overwrite=None, plot_name = None):
    if label_overwrite:
        if len(label_overwrite) == 4:
            label1, label2,label3,label4 = label_overwrite
        elif len(label_overwrite) == 3:
            label1, label2,label3 = label_overwrite
    else:
        label1, label2,label3,label4 = 'State','Ground Truth Host Command','Inferred Command', 'Ground Truth 2 Command'#'Ground Truth', 'Inferred'#'State', 'Command'#,'differences'
    qpos=np.array(qpos_list)
    gt = np.array(gt_list) # ts, dim
    if gt2_list is not None:
        gt2 = np.array(gt2_list) # ts, dim
    infer = np.array(infer_list)
    num_ts, num_dim = gt.shape
    
    if num_dim == 7:
        JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    elif num_dim == 13:
        JOINT_NAMES = ["x", "y", "z", "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]
    STATE_NAMES = JOINT_NAMES + ["gripper"]#6+1=7
    
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):#the real joint angles/states
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()
    # plot arm command
    for dim_idx in range(num_dim):#the action/the desired/the expected joint angles/states
        ax = axs[dim_idx]
        ax.plot(gt[:, dim_idx], label=label2)
        ax.legend()
    # plot arm command
    for dim_idx in range(num_dim):  # the inferred action
        ax = axs[dim_idx]
        ax.plot(infer[:, dim_idx], label=label3)
        ax.legend()
    if gt2_list is not None:
        for dim_idx in range(num_dim):#the action/the desired/the expected joint angles/states NORMALIZED
            ax = axs[dim_idx]
            ax.plot(gt2[:, dim_idx], label=label4)
            ax.legend()
    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)
    plot_path_name = os.path.join(plot_path,plot_name)
    plt.savefig(plot_path_name)
    print(f'Saved qpos plot to: {plot_path_name}')
    plt.close()


def get_image_xht(image_dict, camera_names,t):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(image_dict[cam_name][t], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc_xht_hdf5(config, ckpt_name, camera_names, save_episode=True,dataset_dir=None):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    tf = config['tf']
    if tf == 'joint_angles':
        state_dim = 7
    elif tf == '9d':
        state_dim = 13
    arm_type = config['arm_type']
    if arm_type == 2:
        state_dim*=2
    policy_class = config['policy_class']
    temporal_agg = config['temporal_agg']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    query_frequency = config['chunk_size'] 
    sep_backbones= False  # policy_config['sep_backbones']
    if sep_backbones:
        sepbackbones="sep"
    else:
        sepbackbones="single"
    if temporal_agg:
        query_frequency = 1
        num_queries = config['chunk_size']  
        teornot='te'
    else:
        teornot='non-te'
    files = list()
    for directory in dataset_dir:
        files.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    files = sorted(files)
    episode_id = 0
    for filename in files:
        with h5py.File(filename, 'r') as root:
            if tf == 'joint_angles':
                original_action = root['/action']
                qposgt = root['/observations/qpos']#
            elif tf == '9d':
                original_action_tf = root['/action_tf']
                quat_part_oa = original_action_tf[:,3:7]
                xyz_part_oa = original_action_tf[:,0:3]
                gripper_part_oa = original_action_tf[:,7:8]
                rot_mat_oa = quat_to_rotmtx_batch(quat_part_oa)
                nined_oa = rotmtx_to_9d_batch(rot_mat_oa)
                original_action = np.concatenate((xyz_part_oa,nined_oa,gripper_part_oa), axis=-1)
                qposgt_quat = root['/observations/arm_tf']#
                quat_part = qposgt_quat[:,3:7]
                xyz_part = qposgt_quat[:,0:3]
                gripper_part = qposgt_quat[:,7:8]
                rot_mat = quat_to_rotmtx_batch(quat_part)
                nined = rotmtx_to_9d_batch(rot_mat)
                qposgt = np.concatenate((xyz_part,nined,gripper_part), axis=-1)
            image_dict = dict()
            for cam_name in camera_names:#
                image_dict[cam_name] = root[f'/observations/images/{cam_name}']
            actiongt = original_action
            flaginfer = 0
            ### evaluation loop
            if temporal_agg:
                all_time_actions = torch.zeros([num_queries, num_queries, state_dim]).cuda() 
            qpos_list = []
            target_qpos_list = []
            with torch.inference_mode():
                for t in range(original_action.shape[0]):
                    ### update onscreen render and wait for DT
                    qpos_numpy = qposgt[t]
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    curr_image = get_image_xht(image_dict, camera_names,t)  #
                    ### query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                        if temporal_agg:
                            all_time_actions[t % num_queries] = all_actions  
                            if (t >= num_queries - 1):
                                rowindex = torch.arange(num_queries)
                                columnindex = (torch.arange(t, t - num_queries, -1)) % num_queries
                            else:
                                rowindex = torch.arange(t + 1)
                                columnindex = torch.arange(t, -1, -1)
                            actions_for_curr_step = all_time_actions[rowindex, columnindex]  
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError
                    ### post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action
                    if tf == 'joint_angles':
                        actiongt2i = target_qpos
                    elif tf == '9d':
                        actiongt2i1 = target_qpos[0:3]
                        actiongt2i2 = target_qpos[3:12]
                        actiongt2i3 = target_qpos[12:]
                        actiongt2i2_mat = flat_to_rotmtx_array_batch(actiongt2i2)
                        actiongt2i2_9d = rotmtx_to_9d(actiongt2i2_mat)
                        actiongt2i = np.concatenate((actiongt2i1,actiongt2i2_9d,actiongt2i3), axis = -1) # 
                    if(flaginfer == 0):
                        actioninfer = target_qpos
                        flaginfer = 1
                        actiongt2 = actiongt2i
                    else:
                        actioninfer = np.vstack((actioninfer,target_qpos))
                        actiongt2 = np.vstack((actiongt2,actiongt2i))
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
            print("file name:",filename)
            if tf == 'joint_angles':
                visualize_differences_4(qposgt, actiongt, actioninfer, plot_path = os.path.join(dataset_dir[0], tf + "plot_9d_from_quat","0927-3"), label_overwrite = ['state','actiongt','infer'], plot_name = f'episode_{episode_id}_qpos_{teornot}_qf_{query_frequency}_{sepbackbones}_tf_{tf}.png')
            elif tf == '9d':
                visualize_differences_4(qposgt, actiongt, actioninfer, actiongt2, 
                                      plot_path = os.path.join(dataset_dir[0],tf+"plot_9d_from_quat","0927-merge-4"), label_overwrite = ['state','actiongt','infer','normalized_infer'], plot_name = f'episode_{episode_id}_qpos_{teornot}_qf_{query_frequency}_{sepbackbones}_tf_{tf}.png')
        episode_id += 1

def count_h5_files(dataset_dir):
    # 查找目录下所有的.h5文件
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    # 返回文件数量
    return len(h5_files)
       
def main(args_dict):
    set_seed(args_dict["seed"])
    # command line parameters
    
    dataset_dir = args_dict['dataset_dir']

    num_episodes = 0
    for directory in dataset_dir:
        num_episodes += count_h5_files(directory)

    arm_type = ArmType(args_dict['arm_type'])
    camera_names, qpos_dim, action_dim = get_arm_config(arm_type, args_dict=args_dict)
    
    ckpt_name = f'policy_best.ckpt'
    eval_bc_xht_hdf5(args_dict, ckpt_name, camera_names, save_episode=True,dataset_dir=dataset_dir)  #
        
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy



if __name__ == '__main__':
    parser = get_parser()
    main(vars(parser.parse_args()))

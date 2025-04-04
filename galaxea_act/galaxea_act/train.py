import os
import glob
import torch
import wandb
import numpy as np
import pickle
import h5py
import cv2
os.environ['TORCH_HUB_MIRROR'] = 'https://hf-mirror.com'
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.dataset.episodic_dataset import load_data, load_data_dp, generate_arm_feature_helper
from galaxea_act.utlis.utlis import set_seed, get_arm_config, find_latest_checkpoint, put_text, KDE,hdbscan_with_custom_merge
from galaxea_act.config.constants import IMAGE_WIDTH, IMAGE_HEIGHT, EPISODE_CUTOFF
from galaxea_act.algos.act_policy import ACTPolicy
from galaxea_act.algos.diffusion_policy import DiffusionPolicy
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import transforms

def count_h5_files(dataset_dir):
    # 查找目录下所有的.h5文件
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    # 返回文件数量
    return len(h5_files)

def main(args_dict):
    local_rank = 0
    
    set_seed(args_dict["seed"])
    # command line parameters
    ckpt_dir = args_dict['ckpt_dir']
    with_torso = args_dict['with_torso']
    image_overlay = args_dict['image_overlay']
    with_chassis = args_dict['with_chassis']
    
    dataset_dir = args_dict['dataset_dir']
    split_ratio = args_dict['split_ratio']
    batch_size_train = args_dict['batch_size']
    batch_size_val = args_dict['batch_size']
    num_epochs = args_dict['num_epochs']
    num_episodes = 0
    for directory in dataset_dir:
        num_episodes += count_h5_files(directory)
    use_one_hot_task = args_dict['use_onehot']

    arm_type = ArmType(args_dict['arm_type'])
    camera_names, qpos_dim, action_dim = get_arm_config(arm_type, args_dict = args_dict)
    args_dict['camera_names'] = camera_names
    tf_representation = args_dict["tf"]

    resume = args_dict["resume"]

    policy_class=args_dict['policy_class'] 

    device = torch.device("cuda:{}".format(local_rank))
    
    # label
    if args_dict['label']:
        ckpt_name = 'policy_epoch_399_seed_0.ckpt'
        print(f"Label entropy using {ckpt_dir}/{ckpt_name}")
        label_entropy(args_dict, ckpt_name)
        exit()

    wandb_logger = wandb.init(
        project=args_dict["task_name"],
        group="ACT",  # all runs for the experiment in one group
        config={'dataset_dir': args_dict["dataset_dir"], 'ckpt_dir': args_dict["ckpt_dir"]}
    ) if local_rank == 0 else None
    

    if policy_class == 'ACT':
        load_data_func = load_data
    elif policy_class == 'DP':
        load_data_func = load_data_dp
    # load dataloader
    train_dataloader, val_dataloader, stats, is_sim = load_data_func(dataset_dir, args_dict['chunk_size'], 
                                                                batch_size_train, batch_size_val, camera_names, 
                                                                tf_representation, arm_type, with_torso, with_chassis,
                                                                image_overlay=image_overlay,speedup=args_dict['speedup'],
                                                                teacher_action = args_dict['teacher_action'],use_one_hot_task=use_one_hot_task)
    
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    #load model and move to cuda(local_rank)
    policy = make_policy(policy_class, args_dict)
    device = torch.device("cuda:{}".format(local_rank))
    policy=policy.to(device)
    
    

    loaded_epoch=0
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        print("\033[1;33mTrying to resume \033[0m")
        # raise("Erro confilct to below ckpt, ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')")
        latest_checkpoint, loaded_epoch = find_latest_checkpoint(ckpt_dir)
        if latest_checkpoint is None:            
            print("\033[1;33mNo checkpoint found in the folder.\033[0m")
        else:
            checkpoint_path=os.path.join(ckpt_dir,latest_checkpoint)
            map_location = {"cuda:0": "cuda:{}".format(local_rank)}
            policy.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
            print(f"\033[1;33mLoaded checkpoint from {latest_checkpoint} at epoch {loaded_epoch}\033[0m")

    #load optimization  Need to check more
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_dict["lr"],
                                  weight_decay=args_dict["weight_decay"])
    
    # train loop
    train_bc(train_dataloader, val_dataloader, args_dict, policy, optimizer, loaded_epoch, local_rank, wandb_logger=wandb_logger)

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == "DP":
        # policy_config["num_queries"] = 16        
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, task_emb = data
    data= (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
        task_emb.cuda()
    )
    return policy(data)  # TODO remove None

KDE = KDE()
def label_entropy(config, ckpt_name, save_episode=True):
    set_seed(1)
    ckpt_dir = config["ckpt_dir"]
    policy_class = config["policy_class"]
    camera_names = config["camera_names"]
    temporal_agg = config["temporal_agg"]
    
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    dataset_dir = config["dataset_dir"]
    bags = list()
    for directory in dataset_dir:
        # 对每个文件夹进行递归搜索
        bags.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    bags = sorted(bags)
    trials = []
    for filename in bags:
            # Bookkeeping for all the trials
            h5 = h5py.File(filename, 'r')
            trials.append(h5)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]
    
    query_frequency = 1
    num_queries = config["chunk_size"]
    arm_type = ArmType(config['arm_type'])

    num_rollouts = len(trials)
    
    num_samples = 10
    all_labels = []
    for rollout_id in range(num_rollouts):
        all_left_qpos = trials[rollout_id]["upper_body_observations/left_arm_ee_pose"][()]
        max_timesteps, state_dim = all_left_qpos.shape[0],26 # may increase for real-world tasks
        image_dict = dict()
           

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            ).cuda()
            all_time_samples = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, num_samples,state_dim]
            ).cuda()
            

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []  # for visualization
        
        traj_action_entropy = []
        traj_filter_action = []
        
        with torch.inference_mode():
            for t in tqdm(range(0,max_timesteps-EPISODE_CUTOFF)):
                raw_qpos = generate_arm_feature_helper(trials[rollout_id], arm_type, False, config["tf"] == "9d", t)
                image_dict = dict()
                for cam_name in camera_names:
                    origin_image_bytes = trials[rollout_id][cam_name][t]
                    np_array = np.frombuffer(origin_image_bytes, np.uint8)
                    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)                    
                    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image_dict[cam_name] = img
                all_cam_images = []
                for cam_name in camera_names:
                    all_cam_images.append(image_dict[cam_name])
                obs = {}
                obs['images'] = {'head':all_cam_images[0]}
                all_cam_images = np.stack(all_cam_images, axis=0)
                # construct observations
                image_data = torch.from_numpy(all_cam_images).float()
                qpos_data = torch.from_numpy(raw_qpos).float() 
                qpos_data = pre_process(qpos_data)
                image_data = image_data / 255.0
             
                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  
                image_data = normalize(image_data) 

                ### process previous timestep to get qpos and image_list
                qpos = qpos_data.cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = image_data.cuda().unsqueeze(0)
                             
                ### query policy
                if config["policy_class"] == "ACT"  or config["policy_class"] == "DP":
                    if t % query_frequency == 0:
                        
                        data = curr_image, qpos, None, None, None 
                        action_samples = policy.get_samples(data,num_samples)# ( num_samples, 1，chunk_len,dim)
                        # print(action_samples.shape)
                        _,all_actions = KDE.kde_entropy(action_samples.permute(1,0,2,3).flatten(2))
                        
                        action_samples = action_samples.squeeze().permute(1,0,2) # (chunk_len, num_samples, dim)                       
                        entropy_chunk = torch.mean(torch.std(action_samples,dim=1),dim=-1)
                        raw_action = action_samples.cpu().numpy()
                       
                    
                    entropy = entropy_chunk[t % query_frequency]
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions.reshape(-1,state_dim)
                        all_time_samples[[t], t : t+ num_queries] = action_samples
                            
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        actions_for_next_step = all_time_actions[:, t]
                        samples_populated = torch.all(
                            actions_for_next_step != 0, axis=1
                        )
                        samples_for_curr_step = all_time_samples[:, t]
                        samples_for_curr_step = samples_for_curr_step[samples_populated] 
                        samples_for_curr_step = samples_for_curr_step[:,:,:12]
                        # entropy = torch.log(torch.mean(torch.var(samples_for_curr_step.flatten(0,1),dim=0),dim=-1))
                        entropy,_ = KDE.kde_entropy(samples_for_curr_step.flatten(0,1).unsqueeze(0))
                        # entropy = torch.log(torch.mean(torch.var(action_samples[0],dim=0),dim=-1))
                        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                            
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )

                    traj_action_entropy.append(entropy)
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    raw_action = post_process(raw_action)
                    traj_filter_action.append(raw_action)
                
                else:
                    raise NotImplementedError
                    
                ### store processed image for video 
                entropy_numpy = np.array(traj_action_entropy[-1].cpu())
                store_imgs = {}
                for key, img in obs["images"].items():
                    # img = put_text(img,t,position="bottom")                    
                    store_imgs[key] = put_text(img,entropy_numpy)
                if "images" in obs:
                    image_list.append(store_imgs)
                else:
                    image_list.append({"main": store_imgs})
   
            plt.close()

        # draw trajectory curves
        traj_action_entropy = torch.stack(traj_action_entropy)
        traj_action_entropy = np.array(traj_action_entropy.cpu())
        traj_filter_action = np.stack(traj_filter_action)
        traj_filter_action = np.array(traj_filter_action)

        actions_entropy_norm = traj_action_entropy
        
    
        labels = hdbscan_with_custom_merge(actions_entropy_norm, ckpt_dir, rollout_id)

        image_list = [{'head':put_text(image_list[i]['head'], str(int(labels[i])),  position='bottom')} for i in range(len(image_list))]
        if save_episode :
            os.makedirs(os.path.join(ckpt_dir, f"label"),exist_ok=True)
            save_videos(
                image_list,
                fps=15,
                video_path=os.path.join(ckpt_dir, f"label/rollout{rollout_id}.mp4"),
            )
        
        save_labels = True
        if save_labels:
            trials[rollout_id].close()
            with h5py.File(bags[rollout_id], "r+") as root:
                name = f"/entropy"
                try:
                    root[name] = actions_entropy_norm
                except:
                    del root[name]
                    root[name] = actions_entropy_norm  
            
            with h5py.File(bags[rollout_id], "r+") as root:
                name = f"/labels" # 0 for precision, 1 for non-precision
                try:
                    root[name] = labels
                except:
                    del root[name]
                    root[name] = labels
            
            with h5py.File(bags[rollout_id], "r+") as root:
                name = f"/teacher_action" # 0 for precision, 1 for non-precision
                try:
                    root[name] = traj_filter_action
                except:
                    del root[name]
                    root[name] = traj_filter_action
        all_labels.extend(labels.tolist())
    rate = all_labels.count(0)/len(all_labels)
    print("precision rate: ", rate)



def train_bc(train_dataloader, val_dataloader, config, policy , optimizer, loaded_epoch,local_rank, wandb_logger=None):
    # from time import time
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    split_ratio = config['split_ratio']
    set_seed(seed)
    
    if config['wandb']:
        wandb.init(name=config["run_name"], project=config["task_name"])

    train_history = []
    validation_history = []
    min_val_loss = np.inf

    FINA_VALIDATION=True
    
    def eval():
        with torch.inference_mode():
            policy.eval()
            avg_loss = {}
            eval_count = 0
            for batch_idx, data in enumerate(val_dataloader):
                eval_count += 1
                forward_dict=policy(data)
                for name, value in forward_dict.items():
                    if name not in avg_loss: avg_loss[name] = 0.
                    avg_loss[name] += value.item()
            for name, value in avg_loss.items():
                avg_loss[name] =  avg_loss[name] / eval_count
            if wandb_logger is not None:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                wandb_logger.log({
                    # "eval/loss": loss.item(),
                    "eval/mem": mem}, batch_count)
                for name, value in avg_loss.items():
                    wandb_logger.log({"eval/{}".format(name): value}, batch_count)
                    # if cfg.log_save_image and step % (cfg.log_every * 5) == 0:
                    #     canvas = torch.cat([pixels, colors[..., :3]], dim=2).detach().cpu().numpy()
                    #     canvas = canvas.reshape(-1, *canvas.shape[2:])
                    #     self.run.log({"train/render": wandb.Image(canvas, caption=f"step: {step}")})
    batch_count = 0
    for epoch in tqdm(range(loaded_epoch + 1, num_epochs)):
        rate = epoch / num_epochs
        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))
        policy.train()
        for batch_idx, data in enumerate(train_dataloader):
            batch_count += 1
            optimizer.zero_grad()
            forward_dict=forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            
            if local_rank == 0 and wandb_logger is not None and batch_count % 10 == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                wandb_logger.log({
                    "train/loss": loss.item(),
                    "train/mem": mem}, batch_count)
                for name, value in forward_dict.items():
                    wandb_logger.log({"train/{}".format(name): value.item()}, batch_count)
                # if cfg.log_save_image and step % (cfg.log_every * 5) == 0:
                #     canvas = torch.cat([pixels, colors[..., :3]], dim=2).detach().cpu().numpy()
                #     canvas = canvas.reshape(-1, *canvas.shape[2:])
                #     self.run.log({"train/render": wandb.Image(canvas, caption=f"step: {step}")})

        # save checkpoint and eval
        if (epoch+1) % 50 == 0 and local_rank == 0:
            #save checkpoint or .pth(optinal)
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            # eval()

    if local_rank == 0:
        ckpt_path = os.path.join(ckpt_dir, f'policy_lastest.ckpt')
        torch.save(policy.state_dict(), ckpt_path)
        # eval()



def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, split_ratio):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        log_train_values = np.log(np.array(train_values) + 1e-10)
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), log_train_values, label = 'train')
        if len(validation_history) > 0:
            val_values = [summary[key].item() for summary in validation_history]
            log_val_values = np.log(np.array(val_values) + 1e-10)
            if split_ratio > 0.99:
                plt.plot(np.linspace(0.9 * num_epochs, num_epochs - 1, len(validation_history)), log_val_values, label='validation')
            else:
                plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), log_val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        # plt.close()
    print(f'Saved plots to {ckpt_dir}')

def save_best_info(ckpt_dir, num_epochs, best_epoch, min_val_loss):
    # Append to the same .txt file
    train_info_path = os.path.join(ckpt_dir, 'train_info.txt')
    with open(train_info_path, 'a') as f:
        f.write("Best Epoch and Minimum Validation Loss:\n")  # Add a section header
        f.write(f"Best Epoch until {num_epochs}: {best_epoch}\n")
        f.write(f"Minimum Validation Loss until {num_epochs}: {min_val_loss:.6f}\n")  # Format to 6 decimal places

    print(f"Appended best_epoch and min_val_loss to {train_info_path}")
    
def afterwards(ckpt_dir, best_ckpt_dir):    
    # Loop through files in the source directory
    for file_name in os.listdir(ckpt_dir):
        if file_name.endswith(".png"):  # Check for PNG files
            source_file = os.path.join(ckpt_dir, file_name)
            destination_file = os.path.join(best_ckpt_dir, file_name)
            os.rename(source_file, destination_file)  # Move the file
    # Define the path to the file
    best_until_now_path = os.path.join(ckpt_dir, "policy_best_until_now.ckpt")
    # Check if the file exists and delete it
    if os.path.exists(best_until_now_path):
        os.remove(best_until_now_path)
        print(f"Deleted: {best_until_now_path}")
    else:
        print(f"File does not exist: {best_until_now_path}")

def save_videos(video, fps, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")



if __name__ == '__main__':
    parser = get_parser()
    wandb.setup()
    main(vars(parser.parse_args()))


import os
import cv2
import h5py
import glob
import torch
import numpy as np
import galaxea_act.utlis.utlis as utlis

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from galaxea_act.config.params import ArmType
from galaxea_act.config.constants import TEXT_EMBEDDINGS, TASK_INDEX_MAP, EPISODE_CUTOFF, IMAGE_WIDTH, IMAGE_HEIGHT
from galaxea_act.utlis.utlis import random_overlay, SaltAndPepperNoise
from torch.utils.data.distributed import DistributedSampler

def get_task_emb(filename: str, use_one_hot=False):
    selected_task_index = 22
    for key, value in TASK_INDEX_MAP.items():
        if key in filename:
            selected_task_index = value  
            break
    if use_one_hot:
        return selected_task_index
    else:
        return TEXT_EMBEDDINGS[selected_task_index-22]

def generate_arm_feature_helper(trial: h5py.File, arm_type: ArmType, cmd_flag: bool, task_space_flag: bool, batch_index:int=-1, with_torso=False, with_chassis=False):
    output_features = []
    
    feature_prefix = "upper_body_action_dict/" if cmd_flag else "upper_body_observations/"
    feature_suffix = "_cmd" if cmd_flag else ""
    arm_prefixs = []
    if arm_type in (ArmType.LEFT, ArmType.BIMANUL):
        arm_prefixs.append("left")
    if arm_type in (ArmType.RIGHT, ArmType.BIMANUL):
        arm_prefixs.append("right")

    for arm_prefix in arm_prefixs:
        gripper_name = feature_prefix + f"{arm_prefix}_arm_gripper_position" + feature_suffix
        if task_space_flag:
            pose_name = feature_prefix + f"{arm_prefix}_arm_ee_pose" + feature_suffix
            if batch_index == -1: # batch_mode
                feature_vector = utlis.transform_to_9d_batch(trial[pose_name], trial[gripper_name])
            else:
                feature_vector = utlis.transform_to_9d(trial[pose_name][batch_index], trial[gripper_name][batch_index])
        else:
            pose_name = feature_prefix + f"{arm_prefix}_arm_joint_position" + feature_suffix
            if batch_index == -1:
                feature_vector = np.concatenate([trial[pose_name], trial[gripper_name]], axis=-1)
            else:
                feature_vector = np.concatenate([trial[pose_name][batch_index], trial[gripper_name][batch_index]], axis=-1)
        if with_torso and cmd_flag and (arm_prefix == 'right' or (arm_prefix == 'left' and arm_type == ArmType.LEFT)):#这里的逻辑是？
            pose_name = "lower_body_action_dict/torso_joint_position_cmd"
            if batch_index == -1: # batch_mode
                feature_vector = np.concatenate((feature_vector,trial[pose_name][:,4 - with_torso:4]),axis= -1)  # 3:4 is the waist
            else:
                feature_vector = np.concatenate((feature_vector,trial[pose_name][4 - with_torso:4]),axis= -1)
        if with_chassis and cmd_flag and (arm_prefix == 'right' or (arm_prefix == 'left' and arm_type == ArmType.LEFT)):
            pose_name = "/lower_body_action_dict/chassis_target_speed_cmd"
            if batch_index == -1:  # batch_mode
                feature_vector = np.concatenate((feature_vector, trial[pose_name]), axis=-1)
            else:
                feature_vector = np.concatenate((feature_vector, trial[pose_name]), axis=-1)
        output_features.append(feature_vector)

    if len(output_features) > 1:
        return np.concatenate(output_features, axis=-1)
    else:
        return output_features[0]

def generate_torso_feature_helper(trial: h5py.File, with_torso, batch_index:int=-1):
    output_features = []
    pose_name = "/lower_body_observations/torso_joint_position"
    if batch_index == -1:  # batch_mode
        feature_vector = trial[pose_name][:,4-with_torso:4]
    else:
        feature_vector = trial[pose_name][batch_index][4-with_torso:4]
    output_features.append(feature_vector)
    if len(output_features) > 1:
        return np.concatenate(output_features, axis=-1)
    else:
        return output_features[0]

def generate_chassis_feature_helper(trial: h5py.File, batch_index:int=-1):
    output_features=[]
    feature_name="/lower_body_observations/chassis_joint_position" #待定
    if batch_index == -1:  # batch_mode
        feature_vector = trial[feature_name]
    else:
        feature_vector = trial[feature_name][batch_index]
    output_features.append(feature_vector)
    if len(output_features) > 1:
        return np.concatenate(output_features, axis=-1)
    else:
        return output_features[0]

def downsample_action_with_labels(action, label):
    low_v = 3
    high_v = 6
    horizon, dim = action.shape
    current_action = action
    current_label = label
    indices = []
    i = -1
    while i < horizon:
        if i + high_v < horizon and np.all(current_label[i:i + high_v] == 1):
            i += high_v
            indices.append(i)
        elif i + low_v < horizon:
            i += low_v
            indices.append(i)
        else:
            i = horizon

    new_actions = current_action[indices]
    return new_actions, indices

class EpisodicDatasetGalaxea(torch.utils.data.Dataset):
    def __init__(self, bags_name, dataset_dir, norm_stats, 
                 chunk_size, camera_names, tf_representation, arm_type: ArmType,
                 with_torso, with_chassis, image_overlay, one_hot_emb_flag=False,speedup=False):
        super(EpisodicDatasetGalaxea).__init__()
        self.with_torso = with_torso
        self.with_chassis = with_chassis
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.arm_type = arm_type
        self.trials = []
        self.is_sim = False
        self.speedup= speedup
        self.task_emb_per_trial = []
        self.tf_representation = tf_representation
        self.one_hot_emb_flag = one_hot_emb_flag
        self.image_overlay = image_overlay
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_HEIGHT, IMAGE_WIDTH), scale = (0.8, 1.0)),
            transforms.RandomAffine(degrees = (0, 0), translate = (0.1, 0.1)),
            transforms.ColorJitter(brightness = (0.2,3.0),hue = 0.3),
            transforms.GaussianBlur(kernel_size = (3, 3), sigma = (0.01, 2.)),
            SaltAndPepperNoise(prob_range = (0.0, 0.25)),  # Random noise probability per image
        ])
        if self.image_overlay > 0.000001:
            place365_dir = '/data/places365dataset/val_large'  # '/data/places365dataset/test_large' # '/localdata/ruizihang/places365_standard/train'
            resize_transform = transforms.Compose([
                transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                transforms.ToTensor(),
            ])
            place_dataset = ImageFolder(root = place365_dir, transform = resize_transform)
            self.place_loader = DataLoader(place_dataset, batch_size = 3, shuffle = True)


        for filename in bags_name:
            task_emb = get_task_emb(filename, one_hot_emb_flag)
            # Bookkeeping for all the trials
            h5 = h5py.File(filename, 'r')
            self.trials.append(h5)
            self.task_emb_per_trial.append(task_emb)

        _max_episode_len = 0
        for idx in range(len(self.trials) ):
            trial = self.trials[idx]
            action = generate_arm_feature_helper(trial, self.arm_type, True, self.tf_representation == "9d",with_torso=self.with_torso, with_chassis = self.with_chassis)
            original_action_shape = action.shape
            episode_len = original_action_shape[0] ## cutoff last few
            print(episode_len)
            _max_episode_len = max(_max_episode_len, episode_len)
        self._max_episode_len = _max_episode_len
        print('TOTAL TRIALS = num_episodes = ', len(self.trials), "max :", _max_episode_len)

    def __len__(self):
        return self._max_episode_len * len(self.trials)

    def __getitem__(self, idx, start_ts=None, sample_full_episode = False):
        idx = idx // self._max_episode_len
        trial = self.trials[idx]
        task_emb = self.task_emb_per_trial[idx]
        label = trial['labels']
        action = generate_arm_feature_helper(trial, self.arm_type, True, self.tf_representation == "9d",with_torso=self.with_torso, with_chassis = self.with_chassis)
        original_action_shape = action.shape
        episode_len = original_action_shape[0] ## cutoff last few
        if sample_full_episode:
            start_ts = 0
        elif start_ts is not None:
            start_ts = start_ts
        else:
            start_ts = np.random.choice(episode_len)
        
        # get observation at start_ts only
        qpos = generate_arm_feature_helper(trial, self.arm_type, False, self.tf_representation == "9d", start_ts)
        if self.with_torso:
            qtorso = generate_torso_feature_helper(trial, self.with_torso, start_ts)
            qpos = np.concatenate((qpos,qtorso), axis= -1)
        # if self.with_chassis:
        #     qchassis = generate_chassis_feature_helper(trial, start_ts)
        #     qpos = np.concatenate((qpos, qchassis), axis= -1)
        image_dict = dict()
        for cam_name in self.camera_names:
            origin_image_bytes = trial[cam_name][start_ts]
            np_array = np.frombuffer(origin_image_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
            image_dict[cam_name] = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
        # get all actions after and including start_ts
        action = action[max(0, start_ts - 1):].astype(np.float32) # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned 

        # speedup:
        if self.speedup:
            # action = action[::2]
            label = label[max(0, start_ts - 1):].astype(np.float32)
            action,_ = downsample_action_with_labels(action, label)
            action_len = action.shape[0]

        padded_action = np.zeros((self.chunk_size, original_action_shape[1]), dtype=np.float32) 
        if action_len <= self.chunk_size:
            padded_action[:action_len] = action
        else:
            padded_action[:] = action[:self.chunk_size]
        is_pad = np.zeros(self.chunk_size)
        if action_len < self.chunk_size:
            is_pad[action_len:] = 1  #so is_pad always has dimension 100 

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float() 
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        rand_num = np.random.random()
        if rand_num > 0.5 and (rand_num <= 0.75 or self.image_overlay <= 0.000001):
            # image_data = self.transform(image_data)
            new_obs = image_data / 255.0
            image_data = new_obs.float()
        elif self.image_overlay > 0.000001 and rand_num > 0.75:
            new_obs = image_data / 255.0
            image_data = new_obs.float()
            # image_data = random_overlay(image_data, self.place_loader, self.image_overlay)
        else:
            # normalize image and change dtype to float
            image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if not self.one_hot_emb_flag:
            task_emb = torch.from_numpy(np.asarray(task_emb)).float()
        else:
            task_emb = torch.from_numpy(np.asarray(task_emb)).int()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  
        image_data = normalize(image_data)        


        return image_data, qpos_data, action_data, is_pad, task_emb

    def save_aug_image(self, index, image_data:torch.Tensor):
        image_data = torch.einsum('k c h w  -> k h w c', image_data)
        for i in range(image_data.shape[0]):
            image = image_data[i].detach().cpu().numpy()
            Image.fromarray(image.astype(np.uint8)).save(os.path.join(self.dataset_dir[0], f'saltpepper_img_{index}_camera_{i}.png'))


def get_norm_stats_galaxea(dataset_dir, tf_representation, arm_type, with_torso, with_chassis):
    # files = []
    # for directory in dataset_dir:
    #     files.append()
    files = []
    for directory in dataset_dir:
        abs_directory = os.path.abspath(directory)
        if not os.path.exists(abs_directory):
            print(f"Warning: directory {abs_directory} not exist!")
            continue
        files.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    files = sorted(files)

    all_qpos_data = []
    all_action_data = []
    for filename in files:
        # Check each file to see how many entires it has
        trial = h5py.File(filename, 'r')
        qpos = generate_arm_feature_helper(trial, arm_type, False, tf_representation == "9d")
        action = generate_arm_feature_helper(trial, arm_type, True, tf_representation == "9d",with_torso=with_torso, with_chassis=with_chassis)
        if with_torso:
            qtorso = generate_torso_feature_helper(trial, with_torso)
            qpos = np.concatenate((qpos,qtorso),axis= -1)
        # if with_chassis:
        #     qchassis = generate_chassis_feature_helper(trial)
        #     qpos = np.concatenate((qpos, qchassis), axis= -1)
        all_qpos_data.append(torch.from_numpy(qpos[:-EPISODE_CUTOFF])) 
        all_action_data.append(torch.from_numpy(action[:-EPISODE_CUTOFF]))

    all_qpos_data = torch.concatenate(all_qpos_data, dim=0)  # stack方向？
    all_action_data = torch.concatenate(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data  todo(dongke) 如果是四元数，不需要做正则化？
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0)
    action_std = torch.clip(action_std, 1e-2, 10) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_mean.numpy().astype(np.float32), "action_std": action_std.numpy().astype(np.float32),
             "qpos_mean": qpos_mean.numpy().astype(np.float32), "qpos_std": qpos_std.numpy().astype(np.float32),
             "example_qpos": qpos}

    return stats


def get_minmax_stats_galaxea(dataset_dir, tf_representation, arm_type, with_torso, with_chassis):
    # files = []
    # for directory in dataset_dir:
    #     files.append()
    files = []
    for directory in dataset_dir:
        abs_directory = os.path.abspath(directory)
        if not os.path.exists(abs_directory):
            print(f"Warning: directory {abs_directory} not exist!")
            continue
        files.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    files = sorted(files)

    all_qpos_data = []
    all_action_data = []
    for filename in files:
        # Check each file to see how many entires it has
        trial = h5py.File(filename, 'r')
        qpos = generate_arm_feature_helper(trial, arm_type, False, tf_representation == "9d")
        action = generate_arm_feature_helper(trial, arm_type, True, tf_representation == "9d",with_torso=with_torso, with_chassis=with_chassis)
        if with_torso:
            qtorso = generate_torso_feature_helper(trial, with_torso)
            qpos = np.concatenate((qpos,qtorso),axis= -1)
        # if with_chassis:
        #     qchassis = generate_chassis_feature_helper(trial)
        #     qpos = np.concatenate((qpos, qchassis), axis= -1)
        all_qpos_data.append(torch.from_numpy(qpos[:-EPISODE_CUTOFF])) 
        all_action_data.append(torch.from_numpy(action[:-EPISODE_CUTOFF]))

    all_qpos_data = torch.concatenate(all_qpos_data, dim=0)  # stack方向？
    all_action_data = torch.concatenate(all_action_data, dim=0)
    all_action_data = all_action_data

    # normalize action data  todo(dongke) 如果是四元数，不需要做正则化？
    action_min = all_action_data.min(dim=0)[0]
    action_max = all_action_data.max(dim=0)[0]
    action_std = torch.clip(action_max-action_min, 1e-2) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0)
    qpos_std = torch.clip(qpos_std, 1e-2, 10) # clipping

    stats = {"action_mean": action_min.numpy().astype(np.float32), "action_std": action_std.numpy().astype(np.float32),
             "qpos_mean": qpos_mean.numpy().astype(np.float32), "qpos_std": qpos_std.numpy().astype(np.float32),
             "example_qpos": qpos}

    return stats

def load_data(dataset_dir, chunk_size, batch_size_train, batch_size_val, camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, use_one_hot_task=False, train_ratio=0.8, speedup=False):
    # add new: The selected_episode parameter in the load_data function allows you to specify a particular episode for training or validation.
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_galaxea(dataset_dir,tf_representation, arm_type, with_torso, with_chassis)
    # construct dataset and dataloader
    bags = list()
    for directory in dataset_dir:
        # 对每个文件夹进行递归搜索
        bags.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    bags = sorted(bags)
    shuffled_indices = np.arange(len(bags))
    _index = int(len(shuffled_indices)*train_ratio)
    train_bags = bags[:_index]
    val_bags = bags[_index:]
    train_dataset = EpisodicDatasetGalaxea(train_bags, dataset_dir, norm_stats, chunk_size, 
                                           camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, one_hot_emb_flag=use_one_hot_task, speedup=speedup)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,  pin_memory=True, num_workers=16, prefetch_factor=2)
    
    val_dataset = EpisodicDatasetGalaxea(val_bags, dataset_dir, norm_stats, chunk_size, 
                                         camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, one_hot_emb_flag=use_one_hot_task, speedup=speedup)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val,  pin_memory=True, num_workers=16, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def load_data_dp(dataset_dir, chunk_size, batch_size_train, batch_size_val, camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, use_one_hot_task=False, train_ratio=0.8, speedup=False):
    # add new: The selected_episode parameter in the load_data function allows you to specify a particular episode for training or validation.
    # obtain normalization stats for qpos and action
    norm_stats = get_minmax_stats_galaxea(dataset_dir,tf_representation, arm_type, with_torso, with_chassis)
    # construct dataset and dataloader
    bags = list()
    for directory in dataset_dir:
        # 对每个文件夹进行递归搜索
        bags.extend(glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True))
    bags = sorted(bags)
    shuffled_indices = np.arange(len(bags))
    _index = int(len(shuffled_indices)*train_ratio)
    train_bags = bags[:_index]
    val_bags = bags[_index:]
    train_dataset = EpisodicDatasetGalaxea(train_bags, dataset_dir, norm_stats, chunk_size, 
                                           camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, one_hot_emb_flag=use_one_hot_task, speedup=speedup)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,  pin_memory=True, num_workers=16, prefetch_factor=2)
    
    val_dataset = EpisodicDatasetGalaxea(val_bags, dataset_dir, norm_stats, chunk_size, 
                                         camera_names, tf_representation, arm_type, with_torso, with_chassis, image_overlay, one_hot_emb_flag=use_one_hot_task, speedup=speedup)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val,  pin_memory=True, num_workers=16, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def save_train_info(ckpt_dir, train_indices, val_indices):
    seq_dir = os.path.join(ckpt_dir,'train_info.txt')
    with open(seq_dir, 'w') as f:
        f.write("Train Indices:\n")
        f.write(', '.join(map(str, train_indices)) + '\n')
        f.write("Validation Indices:\n")
        f.write(', '.join(map(str, val_indices)) + '\n')
        print(f"Appended files to {seq_dir}")


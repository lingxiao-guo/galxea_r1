import os
import glob
import torch
import wandb
import numpy as np
import pickle

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from galaxea_act.config.params import ArmType
from galaxea_act.config.parser import get_parser
from galaxea_act.dataset.episodic_dataset import load_data
from galaxea_act.utlis.utlis import compute_dict_mean, set_seed, detach_dict, get_arm_config, find_latest_checkpoint
from galaxea_act.algos.act_policy import ACTPolicy
from galaxea_act.algos.diffusion_policy import DiffusionPolicy
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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

    wandb_logger = wandb.init(
        project=args_dict["task_name"],
        group="ACT",  # all runs for the experiment in one group
        config={'dataset_dir': args_dict["dataset_dir"], 'ckpt_dir': args_dict["ckpt_dir"]}
    ) if local_rank == 0 else None


    # load dataloader
    train_dataloader, val_dataloader, stats, is_sim = load_data(dataset_dir, args_dict['chunk_size'], 
                                                                batch_size_train, batch_size_val, camera_names, 
                                                                tf_representation, arm_type, with_torso, with_chassis,
                                                                image_overlay=image_overlay,
                                                                use_one_hot_task=use_one_hot_task)
    
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
        if (epoch+1) % 100 == 0 and local_rank == 0:
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


if __name__ == '__main__':
    parser = get_parser()
    wandb.setup()
    main(vars(parser.parse_args()))

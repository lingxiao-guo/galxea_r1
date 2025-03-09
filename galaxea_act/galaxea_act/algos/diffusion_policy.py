import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from galaxea_act.models.detr.model_builder import build_ACT_model_and_optimizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch
import hydra
from galaxea_act.config.params import *
from galaxea_act.models.detr.model_builder import *

from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        cfg = OmegaConf.load(args_override['diffusion_policy_cfg'])
        encoder, _ = build_CNNMLP_model_and_optimizer(args_override)
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy,encoder=encoder)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        

    def forward(self, data):
        image, qpos, actions, is_pad , task_emb = data
        env_state = None
        if actions is not None:  # training time
            actions = actions[:, : self.model.horizon]
            is_pad = is_pad[:, : self.model.horizon]
            batch = {}
            batch['action'] = actions
            batch['obs'] = {'qpos':qpos,'image':image}
            raw_loss = self.model.compute_loss(batch)
            
            loss_dict = dict()
            all_l1 = raw_loss
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["loss"] = l1
            return loss_dict
        else:  # inference time
            obs_dict = {'qpos':qpos,'image':image}
            result = self.model.predict_action(obs_dict)
            return result['action_pred']

    def configure_optimizers(self):
        return self.optimizer
    
    def get_samples(self, qpos, image, num_samples=10, actions=None, is_pad=None):
        env_state = None
        
        image = torch.tile(image, (num_samples,1,1,1,1))
        qpos = torch.tile(qpos, (num_samples, 1))
        # inference time        
        obs_dict = {'qpos':qpos,'image':image}
        result = self.model.predict_action(obs_dict)
        return result['action_pred'].unsqueeze(1)
    
    


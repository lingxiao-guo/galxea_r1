import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from galaxea_act.models.detr.model_builder import build_ACT_model_and_optimizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch
from galaxea_act.config.params import *
from galaxea_act.models.detr.model_builder import *



class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super(ACTPolicy, self).__init__()

        # add the params
        self.args=args_override
        args=args_override
        transformer_param = TransformerParams(args["enc_layers"], args["dec_layers"], args["dim_feedforward"], args["hidden_dim"],
            args["dropout"], args["nheads"], pre_norm=args["pre_norm"]
             )
        backbone_param = BackboneParams( args["backbone"], args["lr_backbone"], args["dilation"], args["masks"],
            args["position_embedding"], args["hidden_dim"]
            )
        chunk_size=args["chunk_size"]
        camera_names=args["camera_names"] 
        multi_task=args["multi_task"]
        qpos_dim=args["qpos_dim"]
        action_dim=args["action_dim"]
        use_onehot=args["use_onehot"]
        use_dinov2 = args["backbone"] == "dinov2"
        #model set up
        backbones = []
        use_film = False  # todo(dongke) 默认不启用film方法
        if use_film:
            backbone = build_film_backbone(backbone_param)
        elif use_dinov2:
            backbone = build_dinov2_backbone(backbone_param)
        else:
            backbone = build_backbone(backbone_param)
        backbones.append(backbone)
        if 'upper_body_observations/depth_head' in camera_names:
            depth_backbone = build_backbone(backbone_param)
            backbones.append(depth_backbone)
        # to do: (bye) transformer need to modify
        transformer= build_transformer(transformer_param)
        self.transformer = transformer

        encoder=build_encoder(transformer_param)
        self.encoder = encoder

        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            chunk_size=chunk_size,
            camera_names=camera_names,
            is_multi_task=multi_task,
            qpos_dim=qpos_dim,
            action_dim=action_dim,
            use_film=use_film,
            use_one_hot_task=use_onehot
            )# CVAE decoder

        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')
        self.tf_type = args_override['tf']
        self.mask_rate = 1
    
    def forward(self, data):
        image, qpos, actions, is_pad , task_emb = data
        env_state = None
        # hardcode
        self.mask_rate = 0.75
        mask_num = int(qpos.shape[0] * self.mask_rate)
        # qpos[:mask_num] = 0
        # qpos[:mask_num] = 0
        # hardcode for single task
        task_emb = None
        
        if actions is not None: # training time
            actions = actions[:, :self.model.chunk_size]
            is_pad = is_pad[:, :self.model.chunk_size]

            if task_emb is not None:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, task_emb)
            else:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1_i = (all_l1 * ~is_pad.unsqueeze(-1)).mean(dim=(0,1))  # get the loss in each dimension
            for i in range(l1_i.shape[0]):
                loss_dict[f'l1_{i}'] = l1_i[i]
            l1 = l1_i.mean()  # and the total average loss
            
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            if task_emb is not None:
                a_hat, _, (_, _) = self.model(qpos, image, env_state,task_emb=task_emb) # no action, sample from prior
            else:
                a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior

            return a_hat
        
    def set_mask_rate(self, rate):
        pass

    def get_samples(self, data, num_samples):
        image, qpos, _, _ , task_emb = data
        env_state = None
        # hardcode for single task
        task_emb = None

        if task_emb is not None:
            a_hat, _, (_, _) = self.model.get_samples(qpos, image, env_state,task_emb=task_emb,num_samples=num_samples) # no action, sample from prior
        else:
            a_hat, _, (_, _) = self.model.get_samples(qpos, image, env_state,num_samples=num_samples) # no action, sample from prior

        return a_hat
    
    

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


import torch
import argparse
from torch import nn
from galaxea_act.config.params import *
from galaxea_act.models.detr.detr_vae import DETRVAE, CNNMLP
from galaxea_act.models.detr.backbone import Backbone, Joiner, ResNetFilmBackbone, DINOv2BackBone
from galaxea_act.models.detr.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from galaxea_act.models.detr.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer

def build_position_encoding(pos_emb_type, hidden_dim):
    N_steps = hidden_dim // 2
    if pos_emb_type in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif pos_emb_type in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {pos_emb_type}")

    return position_embedding


def build_transformer(transformer_param: TransformerParams):
    return Transformer(
        d_model=transformer_param.hidden_dim,
        dropout=transformer_param.dropout,
        nhead=transformer_param.nheads,
        dim_feedforward=transformer_param.dim_feedforward,
        num_encoder_layers=transformer_param.enc_layers,
        num_decoder_layers=transformer_param.dec_layers,
        normalize_before=transformer_param.pre_norm,
        return_intermediate_dec=True,
    )


def build_encoder(transformer_param: TransformerParams):
    encoder_layer = TransformerEncoderLayer(transformer_param.hidden_dim, transformer_param.nheads, transformer_param.dim_feedforward,
                                            transformer_param.dropout, transformer_param.activation, transformer_param.pre_norm)
    encoder_norm = nn.LayerNorm(transformer_param.hidden_dim) if transformer_param.pre_norm else None
    encoder = TransformerEncoder(encoder_layer, transformer_param.enc_layers, encoder_norm)

    return encoder


def build_backbone(backbone_param: BackboneParams):
    position_embedding = build_position_encoding(backbone_param.position_embedding, backbone_param.hidden_dim)
    train_backbone = backbone_param.lr_backbone > 0
    return_interm_layers = backbone_param.use_masks
    backbone = Backbone(backbone_param.backbone_name, train_backbone, return_interm_layers, backbone_param.use_dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_dinov2_backbone(backbone_param: BackboneParams):
    print("Using DINOv2 Backbone...")
    position_embedding = build_position_encoding(backbone_param.position_embedding, backbone_param.hidden_dim)
    backbone = DINOv2BackBone()
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_film_backbone(backbone_param: BackboneParams):
    position_embedding = build_position_encoding(backbone_param.position_embedding, backbone_param.hidden_dim)
    film_config = {
        'use': True,
        'use_in_layers': [1, 2, 3],
        'task_embedding_dim': 512,
        'film_planes': [64, 128, 256, 512],
    }
    backbone = ResNetFilmBackbone(backbone_param.backbone_name, film_config=film_config)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_act_model(transformer_param: TransformerParams, backbone_param: BackboneParams,
                    chunk_size, camera_names, is_multi_task, qpos_dim, action_dim, use_one_hot_task):
    backbones = []
    use_film = False  # todo(dongke) 默认不启用film方法
    if use_film:
        backbone = build_film_backbone(backbone_param)
    else:
        backbone = build_backbone(backbone_param)
    backbones.append(backbone)

    if 'upper_body_observations/depth_head' in camera_names:
        depth_backbone = build_backbone(backbone_param)
        backbones.append(depth_backbone)

    transformer = build_transformer(transformer_param)

    encoder = build_encoder(transformer_param)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        chunk_size=chunk_size,
        camera_names=camera_names,
        is_multi_task=is_multi_task,
        qpos_dim=qpos_dim,
        action_dim=action_dim,
        use_film=use_film,
        use_one_hot_task=use_one_hot_task
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_ACT_model_and_optimizer(args):
    transformer_param = TransformerParams(
        args["enc_layers"], args["dec_layers"], args["dim_feedforward"], args["hidden_dim"],
        args["dropout"], args["nheads"], pre_norm=args["pre_norm"]
    )
    backbone_param = BackboneParams(
        args["backbone"], args["lr_backbone"], args["dilation"], args["masks"],
        args["position_embedding"], args["hidden_dim"]
    )

    model = build_act_model(transformer_param, backbone_param,
                            args["chunk_size"], args["camera_names"], 
                            args["multi_task"], args["qpos_dim"],
                            args["action_dim"], args["use_onehot"])

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args["lr_backbone"],
        },
    ]

    # todo(dongke) eval模式下不需要optimizer
    optimizer = torch.optim.AdamW(param_dicts, lr=args["lr"],
                                  weight_decay=args["weight_decay"])

    return model, optimizer



def build_CNNMLP_model_and_optimizer(args):
    
    backbone_param = BackboneParams(
        args["backbone"], args["lr_backbone"], args["dilation"], args["masks"],
        args["position_embedding"], args["hidden_dim"]
    )
    model = build_cnnmlp(args, backbone_param, args["qpos_dim"], args["backbone"]=='dinov2')
    model.cuda()

    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args["lr_backbone"],
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args["lr"], weight_decay=args["weight_decay"]
    )

    return model, optimizer

def build_cnnmlp(args, backbone_param: BackboneParams, qpos_dim, use_dinov2):
    state_dim = qpos_dim  # TODO hardcode
    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args["camera_names"]:
        if use_dinov2:
            backbone = build_dinov2_backbone(backbone_param)
        else:
            backbone = build_backbone(backbone_param)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args["camera_names"],
        use_dinov2=use_dinov2,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
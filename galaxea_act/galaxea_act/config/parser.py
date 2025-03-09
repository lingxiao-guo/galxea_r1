import argparse

def get_parser():
    parser = argparse.ArgumentParser('Set act training parser', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--dataset_dir', nargs='+', help='dataset_dir', required=False)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--arm_type', type=int, help='0: left arm; 1: right arm; 2: both arm', required=True)
    parser.add_argument('--run_name', '--run_name', action='store', type=str, help='run name for logs', required=False)
    parser.add_argument(
        "--diffusion_policy_cfg", action="store", type=str, help="task_name", default='galaxea_act/config/galaxea_diffusion_policy_cnn.yaml'
    )
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=7, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--chunk_size', default=100, type=int,
                        help="Number of action chunk size")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    # for ACT model
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--tf', action='store', type=str, help='joint_angles: joint_angles; 9d: flat rotation matrix;', required=True)
    # add this for multi-task embedding condition
    parser.add_argument('--multi_task', action='store_true')
    parser.add_argument('--use_onehot', action='store_true')
    parser.add_argument('--with_torso', type=int, default=0)
    parser.add_argument('--with_chassis', action='store_true')
    
    #distributed
    #Use --local_rank for argparse if we are going to use torch.distributed.launch to launch distributed training.
    # parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
 
    # other things
    parser.add_argument('--wrc_demo', action='store_true', required=False)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--start_flag', action='store_false', required=False)
    parser.add_argument('--image_overlay', action='store', default=0.00, type=float, help='image_overlay ratio, if 0.0, then means disabled', required=False)
    parser.add_argument('--split_ratio', action='store', default=1.00, type=float, help='split ratio, if 1.0, then means train is validation', required=False)
    return parser

from enum import Enum
from dataclasses import dataclass


@dataclass
class TransformerParams:
    enc_layers: int = 4  # Number of encoding layers in the transformer
    dec_layers: int = 6  # Number of decoding layers in the transformer
    dim_feedforward: int = 2048  # Intermediate size of the feedforward layers in the transformer blocks
    hidden_dim: int = 256  # Size of the embeddings (dimension of the transformer)
    dropout: float = 0.1  # Dropout applied in the transformer
    nheads: int = 8  # Number of attention heads inside the transformer's attentions
    activation: str = "relu"
    pre_norm: bool = False

@dataclass
class BackboneParams:
    backbone_name: str = "resnet18"  # Name of the convolutional backbone to use
    lr_backbone: float = 1e-5
    use_dilation: bool = False  # If true, we replace stride with dilation in the last convolutional block (DC5)
    use_masks: bool = False  # Train segmentation head if the flag is provided
    position_embedding: str = "sine"  # Type of positional embedding to use on top of the image features
    hidden_dim: int = 256  # Size of the embeddings (dimension of the transformer), for position embedding

@dataclass
class InferRosNodeParam:
    msg_time_max_diff:float = 0.1  # 不同消息之间最大能容忍的时间差，单位为秒。超过这个时间差的话，目前只会抛出报警
    image_width:int = 320
    image_height:int = 240
    head_camera_topic:str = "/zed2/zed_node/rgb_raw/image_raw_color/compressed" # "/camera_head/color/image_raw/compressed" 
    hand_camera_topic:str = "/camera_right_hand/color/image_raw/compressed"
    joint_states_topic:str = "/a1_robot_right/joint_states"
    arm_command_topic:str = "a1_robot_right/arm_command" 

class ArmType(Enum):
    LEFT = 0
    RIGHT = 1
    BIMANUL = 2

    def __str__(self):
        return f"{self.name}"
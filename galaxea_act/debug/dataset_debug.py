import h5py
import imageio
import numpy as np
from matplotlib import pyplot as plt

def saveimg_helper(input_image, output_path):
    image = input_image.astype(np.uint8)
    print(image.shape)
    # image = rearrange(image, 'c h w -> h w c')
    print("image shape: ", image.shape)
    imageio.imwrite(output_path, image)

def load_h5_data(h5_path, jn_flag=False):
    trial = h5py.File(h5_path, 'r')
    
    qpos = np.array(trial['observations']['qpos'], dtype=np.float32)
    if jn_flag:
        action = np.array(trial['action_host'], dtype=np.float32)
        head_img = np.array(trial['observations']['images']['body'])[300]
        hand_img = np.array(trial['observations']['images']['righthand'])[300]
        saveimg_helper(head_img, "head_img_jn.png")
        saveimg_helper(hand_img, "hand_img_jn.png")
    else:
        action = np.array(trial['action'], dtype=np.float32)
        head_img = np.array(trial['observations']['images']['rgb_head'])[300]
        hand_img = np.array(trial['observations']['images']['rgb_hand'])[300]
        saveimg_helper(head_img, "head_img_mt.png")
        saveimg_helper(hand_img, "hand_img_mt.png")

    
    return qpos, action


def plot(gripper_pos_list, legend_list, title):
    plt.figure()
    color = ['g', 'b', 'r']
    for i in range(len(gripper_pos_list)):
        y = gripper_pos_list[i]
        x = range(len(y))
        plt.plot(x, y, color[i], label=legend_list[i])
    
    plt.grid()
    plt.legend()
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title(title)
    plt.savefig(f"{title}.png")


def main(jn_path, mt_path):
    _, action_jn = load_h5_data(jn_path, True)
    _, action_mt = load_h5_data(mt_path)
    for i in range(7):
        gripper_pos_list = [
            action_jn[:, i],
            action_mt[:, i]
        ]
        legend_list = ["jn_version", "mt_version"]
        plot(gripper_pos_list, legend_list, f"action_index_{i}")


if __name__ == "__main__":
    import sys
    jn_path = sys.argv[1]
    mt_path = sys.argv[2]
    main(jn_path, mt_path)
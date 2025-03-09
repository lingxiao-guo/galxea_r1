import numpy as np
import h5py
import glob
# import plotly.graph_objects as go
import argparse


def register_quat(episode, name):
    a = episode[name]
    # Convert to numpy array
    a_np = np.array(a)
    quat = a_np[:, 3: 7]
    
    quat_couple_number = 2
    dot = np.diag(np.dot(quat[0: -1], quat[1:].T))
    # fig = go.Figure([go.Scatter(y=quat[:, i], name=name) for i, name in enumerate(["x", "y", "z", "w"])])
    # fig.add_trace(go.Scatter(x=np.arange(1, len(quat)), y=dot, name="dot"))
    # fig.show()
    
    shift_indices = np.where(dot < 0)[0] + 1
    if len(shift_indices) % quat_couple_number == 1:
        shift_indices = np.append(shift_indices, len(quat))
    fixed_quat = quat.copy()
    for i in range(len(shift_indices) // quat_couple_number):
        fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]] = -fixed_quat[shift_indices[2 * i]: shift_indices[2 * i + 1]]
    assert fixed_quat.shape == quat.shape
    # Replace the original quaternion data with fixed_quat
    a_np[:, 3: 7] = fixed_quat
    a[...] = a_np
    # fig = go.Figure([go.Scatter(y=fixed_quat[:, i], name=name) for i, name in enumerate(["x", "y", "z", "w"])])
    # fig.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split .h5 files into training and testing sets.")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing .h5 files"
    )
    args = parser.parse_args()
    
    h5_path = args.folder_path
    
    episode_paths = glob.glob(f"{h5_path}/**/*.h5", recursive=True)
    episode_paths.sort()
    assert len(episode_paths) > 0, "No .h5 files found in the specified folder."
    print(len(episode_paths))
    for i in range(len(episode_paths)):
        episode = h5py.File(episode_paths[i], "r+")
        register_quat(episode, "upper_body_action_dict/left_arm_ee_pose_cmd")
        register_quat(episode, "upper_body_action_dict/right_arm_ee_pose_cmd")
        print("Episode", i, "done.")
        episode.close()
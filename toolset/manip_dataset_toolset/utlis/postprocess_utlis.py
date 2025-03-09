import cv2
import rosbag
import numpy as np
from enum import Enum

from scipy.interpolate import interp1d
from pyquaternion import Quaternion as pyQuaternion  # pyQuaternion use the seqence: w, x, y, z

class ArmType(Enum):
    LEFT = 0
    RIGHT = 1
    BIMANUL = 2

    def __str__(self):
        return f"{self.name}"

def interpolate_1d(target_time_steps: np.ndarray, source_timestamps: np.ndarray, source_values: np.ndarray):
    noise_topic_threshold = 0.2 #filter out the noisy topic whose number is less than 20% of the target topic
    if len(source_timestamps) < noise_topic_threshold * len(target_time_steps):
        return source_values

    # Create the interpolation function for all columns simultaneously
    f = interp1d(source_timestamps, source_values, kind='linear', axis=0, fill_value="extrapolate")

    # Perform the interpolation
    interpolated_states = f(target_time_steps)

    return interpolated_states


def interpolate_transform(target_time_stamps, source_time_stamps, source_values):
    positions = source_values[:, :3]
    quaternions = [pyQuaternion(source_values[i, 6], source_values[i, 3], source_values[i, 4], source_values[i, 5]) for i in range(source_values.shape[0])]

    target_values = np.empty((len(target_time_stamps), 7))

    # Use numpy's searchsorted to find indices for interpolation
    indices = np.searchsorted(source_time_stamps, target_time_stamps, side='right')

    # Ensure indices are within valid range
    indices = np.clip(indices, 1, len(source_time_stamps) - 1)

    t0 = source_time_stamps[indices - 1]
    t1 = source_time_stamps[indices]

    pos0 = positions[indices - 1]
    pos1 = positions[indices]

    quat0 = np.array([quaternions[i - 1] for i in indices])
    quat1 = np.array([quaternions[i] for i in indices])

    # Calculate interpolation factors
    t_interp = (target_time_stamps - t0) / (t1 - t0)

    target_values[:, :3] = pos0 + np.expand_dims(t_interp, axis=1) * (pos1 - pos0)
    
    interpolated_quats = [pyQuaternion.slerp(quat0[i], quat1[i], t_interp[i]).normalised for i in range(len(t_interp))]
    target_values[:, 3:] = np.array([[q.x, q.y, q.z, q.w] for q in interpolated_quats])

    # For timestamps out of the source range
    target_values[target_time_stamps <= source_time_stamps[0], :3] = positions[0]
    target_values[target_time_stamps <= source_time_stamps[0], 3:] = [quaternions[0].x, quaternions[0].y, quaternions[0].z, quaternions[0].w]
    
    target_values[target_time_stamps >= source_time_stamps[-1], :3] = positions[-1]
    target_values[target_time_stamps >= source_time_stamps[-1], 3:] = [quaternions[-1].x, quaternions[-1].y, quaternions[-1].z, quaternions[-1].w]

    return target_values

def frequency_helper(timestamps, target_topic, logger=None):
    if len(timestamps) > 1:
        lasting_time = timestamps[-1] - timestamps[0]
        frequency = len(timestamps) / lasting_time
        print(f"read {len(timestamps)} arm command messages from topic {target_topic} approximate frequency: {frequency:.3f}")
        if logger is not None:
            logger.info(f"read {len(timestamps)} arm command messages from topic {target_topic} approximate frequency: {frequency:.3f}")
    else:
        print(f"warning {target_topic} has no message")
        if logger is not None:
            logger.warn(f"warning {target_topic} has no message")

def load_joint_state(input_bag:rosbag.Bag, target_topic:str, logger):
    timestamps = []
    positions = []
    velocities = []
    for topic, msg, t in input_bag.read_messages(topics=target_topic):
        # Append the timestamp and positions to the lists
        timestamps.append(t.to_sec())
        positions.append(list(msg.position))
        velocities.append(list(msg.velocity))
    frequency_helper(timestamps, target_topic, logger)

    timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
    positions = np.array(positions)
    velocities = np.array(velocities)

    return timestamps, positions, velocities

def load_twist_state(input_bag:rosbag.Bag, target_topic:str, logger):
    timestamps = []
    velocities = []
    for topic, msg, t in input_bag.read_messages(topics=target_topic):
        # Append the timestamp and positions to the lists
        timestamps.append(t.to_sec())
        velocities.append([msg.linear.x, msg.linear.y, msg.angular.z])
    frequency_helper(timestamps, target_topic, logger)

    timestamps = np.array(timestamps)  # do not set dtype for timestamps, it exceeds the upper bound of fp32
    velocities = np.array(velocities)

    return timestamps, velocities

def compress_image_to_bytes(image_array, extension='png'):
    # Encode the image
    success, encoded_image = cv2.imencode(f'.{extension}', image_array)
    if not success:
        raise Exception("Image encoding failed!")
    
    # Convert to bytes
    return encoded_image.tobytes()


def registrated_images(target_timestamps, source_timestamps, source_images):
    """
    Find the indices of the closest values in `reference_times` for each value in `target_times`.
    """
    closest_indices = np.abs(source_timestamps[:, None] - target_timestamps).argmin(axis=0)
    # breakpoint()
    registrated_hand_timestamps = source_timestamps[closest_indices]
    source_images = np.array(source_images)
    registrated_hand_images=source_images[closest_indices]
    return registrated_hand_timestamps, registrated_hand_images


def dict_to_dict_list(dictionary, index_array):
    dict_list = []
    for i in range(len(index_array)-1):
        dict_i = {}
        for key in dictionary.keys():
            d_key_i = dictionary[key]
            dict_i[key]=d_key_i[index_array[i]:index_array[i+1]]
        dict_list.append(dict_i)
    return dict_list
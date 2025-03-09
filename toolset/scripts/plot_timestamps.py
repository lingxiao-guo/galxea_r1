import numpy as np


if __name__ == "__main__":
    # Load the timestamps from the .npz file
    timestamps = np.load("aligned_timestamps.npz")
    reference_timestamps = timestamps["reference_timestamps"]
    aligned_timestamps_left_hand_images = timestamps["left_hand_timestamps"]
    aligned_timestamps_right_hand_images = timestamps["right_hand_timestamps"]
    print("Reference timestamps length:", len(reference_timestamps))
    print("Aligned left hand timestamps length:", len(aligned_timestamps_left_hand_images))
    print("Aligned right hand timestamps length:", len(aligned_timestamps_right_hand_images))
    # calculate the difference between the reference timestamps and the aligned timestamps
    left_hand_diff = np.abs(reference_timestamps - aligned_timestamps_left_hand_images)
    right_hand_diff = np.abs(reference_timestamps - aligned_timestamps_right_hand_images)
    reference_diff = np.array(reference_timestamps - reference_timestamps)
    # plot the difference between the reference timestamps and the aligned timestamps
    import matplotlib.pyplot as plt
    plt.plot(left_hand_diff[0:500], label="Left Hand")
    plt.plot(right_hand_diff[0:500], label="Right Hand")
    plt.plot(reference_diff[0:500], label="Reference Timestamps")
    plt.legend()
    plt.xlabel("Index")
    plt.ylabel("Difference(s)")
    plt.title("Difference between Reference Timestamps and Aligned Timestamps")
    plt.show()
    plt.savefig("difference_plot.png")    
 
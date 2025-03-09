import sys
import h5py
import numpy as np

def calculate_average_error(file1_path, file2_path, field_path="lower_body_observations/floating_base_pose"):
    # Open the HDF5 files
    with h5py.File(file1_path, 'r') as file1, h5py.File(file2_path, 'r') as file2:
        # Read the right_arm_ee_pose data from both files
        data1 = np.array(file1[field_path])
        data2 = np.array(file2[field_path])
        
        # Check if both datasets have the same shape
        if data1.shape != data2.shape:
            raise ValueError("The two datasets must have the same shape for comparison.")
        
        # Calculate the absolute error for each time step and dimension
        error = np.abs(data1 - data2)
        
        # Average the error across the time_steps dimension
        average_error = np.mean(error, axis=0)
        
        return average_error

# Example usage


file1_path = sys.argv[1]
file2_path = sys.argv[2]
average_error = calculate_average_error(file1_path, file2_path)

print("Average Error for each dimension:", average_error)
print("Overall Average Error:", np.mean(average_error))

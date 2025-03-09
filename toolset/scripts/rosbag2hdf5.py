import argparse
import manip_dataset_toolset.utlis.postprocess_utlis as utlis
from manip_dataset_toolset.post_process.process_rosbag import RosbagProcessor

# Create the argument parser
parser = argparse.ArgumentParser(description="Process ROS bag files")

# Only required arguments
parser.add_argument("rosbag_dir", type=str, help="Directory containing ROS bag files")
parser.add_argument("output_dir", type=str, help="Output directory for processed files")
parser.add_argument("task_id", type=str, help="The task id for processed files")

# Optional arguments with default values (they are not required)
parser.add_argument("--arm_type", type=int, choices=[0, 1, 2], default=2, help="Arm type (0 for left arm, 1 for right arm, 2 for both arms)")
parser.add_argument("--id_interval", type=int, default=1, help="The interval between the ids of the different tasks. 0 means single task")
parser.add_argument("--h5_start_id", type=int, default=0, help="Starting ID for HDF5 files (default: 0)")
parser.add_argument("--log_file_dir", type=str, default="log/", help="Directory for log files (default: 'log/')")
parser.add_argument("--task_space_cmd", action='store_true', default=False, help="True for task-space control, False for joint-space control")
parser.add_argument("--zarr", action='store_true', help="True for generating zarr files")
parser.add_argument("--num_parallel", type=int, default=40, help="Number of parallel processes for processing the data")

# Parse the arguments
args = parser.parse_args()

# Convert arm type to ArmType enum
arm_type = utlis.ArmType(args.arm_type)

# Initialize the RosbagProcessor
bag_processor = RosbagProcessor(arm_type, args.task_space_cmd, args.log_file_dir, args.zarr, args.num_parallel)

# Process the directory with the provided required arguments and optional arguments
bag_processor.process_dir(args.rosbag_dir, args.output_dir, args.task_id, 
                          debug_info=False, start_id=args.h5_start_id, id_interval=args.id_interval)

import os
import re
import sys

def rename_files_in_directory(directory_path: str, new_task_id: str):
    # 检查新task_id的格式是否为五位数字
    if not re.match(r"^\d{5}$", new_task_id):
        print("Error: The new task_id must be exactly five digits.")
        return

    # 确保目录存在
    if not os.path.isdir(directory_path):
        print(f"Error: The directory {directory_path} does not exist.")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 仅处理以 .bag 或 .yaml 结尾的文件
        if filename.endswith(('.bag', '.yaml', 'h5')):
            # 使用正则表达式匹配文件名格式
            match = re.match(r"(\d{5})-(\d{4}-\d{14})\.(bag|yaml|h5)", filename)
            if match:
                # 获取文件的后半部分，即不包含前五位task_id的部分
                rest_of_filename = match.group(2)
                file_extension = match.group(3)
                
                # 构建新的文件名，替换前五位为新的task_id
                new_filename = f"{new_task_id}-{rest_of_filename}.{file_extension}"
                
                # 获取完整的文件路径
                old_file_path = os.path.join(directory_path, filename)
                new_file_path = os.path.join(directory_path, new_filename)
                
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {filename} to {new_filename}")
            else:
                print(f"Skipping file {filename}, does not match expected pattern.")
                
# 检查是否提供了足够的命令行参数
if len(sys.argv) != 3:
    print("Usage: python script.py <directory_path> <new_task_id>")
else:
    directory_path = sys.argv[1]
    new_task_id = sys.argv[2]
    rename_files_in_directory(directory_path, new_task_id)

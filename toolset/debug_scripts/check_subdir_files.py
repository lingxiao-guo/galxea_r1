import os
import warnings

def count_files(directory):
    """
    遍历指定目录，统计每个文件夹中的.bag和.yaml文件数量，并警告其他类型的文件。
    """
    for root, dirs, files in os.walk(directory):
        # 初始化计数器
        bag_count = 0
        yaml_count = 0
        other_files = 0

        # 遍历文件
        for file in files:
            if file.endswith('.bag'):
                bag_count += 1
            elif file.endswith('.yaml'):
                yaml_count += 1
            else:
                other_files += 1

        # 打印统计结果
        print(f"文件夹：{root}")
        print(f".bag文件数量：{bag_count}")
        print(f".yaml文件数量：{yaml_count}")

        # 如果有其他类型的文件，发出警告
        if other_files > 0:
            warnings.warn(f"在文件夹{root}中检测到{other_files}个其他类型的文件。")

if __name__ == "__main__":
    # 用户输入路径
    path = input("请输入要统计的文件夹路径：")
    # 调用函数
    count_files(path)
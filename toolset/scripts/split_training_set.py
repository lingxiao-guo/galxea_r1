import os
import random
import argparse

def split_dataset(folder_path):
    # 获取所有 .h5 文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    # 随机打乱文件顺序
    random.shuffle(all_files)
    
    # 按 80% 和 20% 分割
    split_index = int(len(all_files) * 0.8)
    training_files = all_files[:split_index]
    testing_files = all_files[split_index:]
    
    # 创建子目录映射而非实际文件复制
    training_dir = os.path.join(folder_path, "training_set")
    testing_dir = os.path.join(folder_path, "testing_set")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(testing_dir, exist_ok=True)
    
    # 记录训练集和测试集文件路径
    training_paths = [os.path.join(folder_path, file) for file in training_files]
    testing_paths = [os.path.join(folder_path, file) for file in testing_files]
    
    print(f"分割完成：\n训练集文件数：{len(training_files)}\n测试集文件数：{len(testing_files)}")
    print(f"训练集保存路径: {training_dir}\n测试集保存路径: {testing_dir}")
    return training_paths, testing_paths, training_dir, testing_dir

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Split .h5 files into training and testing sets.")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing .h5 files"
    )
    parser.add_argument(
        "--no-move",
        action="store_false",
        default=True,
        dest="move",
        help="Do not move files to subdirectories (default: move files)"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 执行分割
    training_paths, testing_paths, training_dir, testing_dir = split_dataset(args.folder_path)
    
    if args.move:
        for file in training_paths:
            os.rename(file, os.path.join(training_dir, os.path.basename(file)))
        for file in testing_paths:
            os.rename(file, os.path.join(testing_dir, os.path.basename(file)))
        print("所有文件已移动到对应子目录中。")
    else:
        print("文件未移动，只记录路径映射。")


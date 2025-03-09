import os
import sys
import yaml
from collections import OrderedDict

# 自定义的OrderedDumper以确保YAML文件中的键不被排序
class OrderedDumper(yaml.SafeDumper):
    pass

def yaml_represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())
yaml.add_representer(OrderedDict, yaml_represent_ordereddict, Dumper=OrderedDumper)

def extract_info_from_filename(filename):
    # 从文件名中提取 task_id、task_trajectory_id 和 episode_id
    parts = filename.rsplit('.', 1)[0].split('-')  # 移除扩展名并按-分割
    if len(parts) >= 3:
        task_id = parts[0]
        task_trajectory_id = parts[1]
        episode_id = '-'.join(parts[:3])  # 格式为 task_id-task_trajectory_id-timestamp
        return task_id, task_trajectory_id, episode_id
    return None, None, None

def update_yaml_files(new_yaml_path, existing_yaml_dir):
    # 读取新配置的yaml文件
    with open(new_yaml_path, 'r', encoding='utf-8') as new_file:
        new_config = yaml.safe_load(new_file)
    
    # 遍历existing_yaml_dir目录下的每个yaml文件
    for filename in os.listdir(existing_yaml_dir):
        file_path = os.path.join(existing_yaml_dir, filename)
        
        # 检查是否是YAML文件
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(file_path, 'r', encoding='utf-8') as existing_file:
                existing_data = yaml.safe_load(existing_file)

            # 从文件名中提取 task_id、task_trajectory_id 和 episode_id
            task_id, task_trajectory_id, episode_id = extract_info_from_filename(filename)
            collection_timestamp = existing_data.get('collection_timestamp')
            
            # 创建更新后的数据字典
            updated_data = OrderedDict()
            task_id_added = False

            # 将新配置中的键值插入到新字典中，且在task_id之后插入提取的信息
            for key, value in new_config.items():
                updated_data[key] = value
                if key == 'task_id' and not task_id_added:
                    # 在task_id后添加提取的信息
                    updated_data['task_trajectory_id'] = task_trajectory_id
                    updated_data['collection_timestamp'] = collection_timestamp
                    updated_data['episode_id'] = episode_id  # 使用文件名中的 episode_id
                    task_id_added = True
            
            # 保存覆盖旧的yaml文件
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(updated_data, file, Dumper=OrderedDumper, default_flow_style=None, width=1000, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    # 从命令行参数中获取new_yaml_path和existing_yaml_dir
    if len(sys.argv) < 3:
        print("Usage: python script.py <new_yaml_path> <existing_yaml_dir>")
        sys.exit(1)
    
    new_yaml_path = sys.argv[1]
    existing_yaml_dir = sys.argv[2]
    
    update_yaml_files(new_yaml_path, existing_yaml_dir)

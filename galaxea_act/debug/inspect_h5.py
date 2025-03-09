import h5py

def print_h5_shapes(name, obj):
    """
    打印 HDF5 文件中数据集的名称和尺寸
    """
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}")

def list_h5_fields(file_path):
    """
    列出 HDF5 文件中所有数据集的名称和尺寸，包括嵌套的字段
    """
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_h5_shapes)


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    list_h5_fields(file_path)
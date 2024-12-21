import os
from pathlib import Path
import re


class ConfigDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        # 返回字典中对应的值
        if item in self:
            return self[item]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")


def create_next_train_directory(base_dir, prefix='train'):
    os.makedirs(base_dir, exist_ok=True)
    # 用于匹配类似 "train1", "train2", "train10" 的目录名
    pattern = fr"^{prefix}(\d+)$"

    # 获取当前目录下所有的子目录名
    dirs = os.listdir(base_dir)

    # 存储符合 "trainx" 模式的数字
    train_nums = []

    for dir_name in dirs:
        match = re.match(pattern, dir_name)
        if match:
            # 提取数字部分并添加到列表中
            train_nums.append(int(match.group(1)))

    # 如果找到了符合条件的目录
    if train_nums:
        # 获取最大的数字并生成下一个目录名
        max_train_num = max(train_nums)
        new_train_dir = f"{prefix}{max_train_num + 1}"
    else:
        # 如果没有找到符合条件的目录，创建 "train1"
        new_train_dir = f"{prefix}1"
    # 生成新目录的完整路径
    new_train_path = os.path.join(base_dir, new_train_dir)

    # 创建新目录
    try:
        os.makedirs(new_train_path, exist_ok=True)
        print(f"Created new directory: {new_train_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")
    return new_train_path
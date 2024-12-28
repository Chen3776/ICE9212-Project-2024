import os
import pandas as pd
import shutil

def create_new_structure(image_dir: str, new_image_dir: str):
    # 创建新的文件夹结构
    train_dir = os.path.join(new_image_dir, 'train')
    val_dir = os.path.join(new_image_dir, 'val')
    test_dir = os.path.join(new_image_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 读取 CSV 文件
    train_csv_path = os.path.join(image_dir, 'train.csv')
    val_csv_path = os.path.join(image_dir, 'val.csv')

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # 处理训练数据
    for _, row in train_df.iterrows():
        image_path = row['image:FILE']
        label = row['label:LABEL']
        new_image_path = os.path.join(train_dir, str(label), os.path.basename(image_path))
        
        # 创建标签子文件夹
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        
        # 移动图片
        shutil.move('code/datafiles/caltech-101/' + image_path, new_image_path)

    # 处理验证数据
    for _, row in val_df.iterrows():
        image_path = row['image:FILE']
        label = row['label:LABEL']
        new_image_path = os.path.join(val_dir, str(label), os.path.basename(image_path))
        
        # 创建标签子文件夹
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        
        # 移动图片
        shutil.move('code/datafiles/caltech-101/' + image_path, new_image_path)


    # 处理测试数据
    test_data_dir = '/remote-home/pengyichen/diffusion/code/datafiles/caltech-101/caltech-101/test'  # 测试集原路径
    if not os.path.exists(test_data_dir):
        print("Test data directory does not exist.")
        return

    for entry in os.scandir(test_data_dir):
        if entry.is_dir() and len(entry.name) >= 3:
            label = entry.name[1:3]  # 标签为文件夹名的前三个字符
            for image_name in os.listdir(entry.path):
                image_path = os.path.join(entry.path, image_name)
                new_image_path = os.path.join(test_dir, str(label), image_name)
                
                # 创建标签子文件夹
                os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
                
                # 移动图片
                shutil.move(image_path, new_image_path)

# 调用函数
image_dir = 'code/datafiles/caltech-101'
new_image_dir = 'code/datafiles/new_caltech-101'
create_new_structure(image_dir, new_image_dir)
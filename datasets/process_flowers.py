import os
import pandas as pd
import shutil

def create_new_structure(image_dir: str, new_image_dir: str):

    test_dir = os.path.join(new_image_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    test_csv_path = '/remote-home/pengyichen/diffusion/code/datafiles/flowers-102/sample_submission.csv'
    test_df = pd.read_csv(test_csv_path)
    
    test_data_dir = '/remote-home/pengyichen/diffusion/code/datafiles/flowers-102/test'
    if not os.path.exists(test_data_dir):
        print("Test data directory does not exist.")
        return

    for _, row in test_df.iterrows():
        image_name = row[0]  # 文件名是第一列
        label = row[1]  # 标签是第二列
        old_image_path = os.path.join(test_data_dir, image_name)
        new_image_path = os.path.join(test_dir, str(label), image_name)
        
        # 创建标签子文件夹
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        
        # 移动图片
        shutil.move(old_image_path, new_image_path)

# 调用函数
image_dir = '/remote-home/pengyichen/diffusion/code/datafiles/flowers102'
new_image_dir = '/remote-home/pengyichen/diffusion/code/datafiles/new-flowers102'
create_new_structure(image_dir, new_image_dir)
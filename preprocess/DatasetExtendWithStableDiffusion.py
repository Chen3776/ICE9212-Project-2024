import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before the diffusers

from pathlib import Path
import random
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import json

class DataAugmentation:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to(self.device)
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32).to(self.device)
        json_file_path = '/remote-home/pengyichen/diffusion/code/datafiles/caltech-101/cat_to_name.json'
        with open(json_file_path, 'r') as file:
            self.label_to_name = json.load(file)

    def generate_image(self, input_image_path: str, output_class_dir: str, iteration: int, class_name: str):
        image = Image.open(input_image_path)
        prompt = f"Generate a photo of {class_name}"
        generated_image = self.pipe(prompt=prompt, image=image, quiet=True).images[0]
        output_path = os.path.join(output_class_dir, f"generated_{iteration}.png")
        generated_image.save(output_path)

    def get_random_class(self, mode_dir: str):
        # 获取所有类别的名称列表
        class_names = [name for name in os.listdir(mode_dir) 
                       if os.path.isdir(os.path.join(mode_dir, name))]
        
        # 随机选择一个类别
        selected_class_name = random.choice(class_names)
        return selected_class_name

    def augment_dataset(self, mode: str, k: int):
        mode_dir = os.path.join(self.data_dir, mode)
        output_mode_dir = os.path.join(self.output_dir, f"diffusion_{mode}")
        os.makedirs(output_mode_dir, exist_ok=True)

        for i in tqdm(range(k), desc="Generating images"):
            # 直接随机选择一个类别
            selected_str_label = self.get_random_class(mode_dir)
            selected_class_name = self.label_to_name.get(selected_str_label)
            
            output_class_dir = os.path.join(output_mode_dir, selected_str_label)
            os.makedirs(output_class_dir, exist_ok=True)

            class_dir = os.path.join(mode_dir, selected_str_label)
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            selected_image_file = random.choice(image_files)
            selected_image_path = os.path.join(class_dir, selected_image_file)
            
            # 为选定的图片生成一张新图片
            self.generate_image(selected_image_path, output_class_dir, i, selected_class_name)

if __name__ == "__main__":
    
    data_dir = '/remote-home/pengyichen/diffusion/code/datafiles/caltech-101'
    output_dir = '/remote-home/pengyichen/diffusion/code/datafiles/generated_caltech-101'

    augmentation = DataAugmentation(data_dir, output_dir)
    k = 500  # 超参数k作为生成图片的数目

    for mode in ['train', 'valid', 'test']:
        augmentation.augment_dataset(mode, k)

    print(f"Data augmentation completed. New images saved to {output_dir}")
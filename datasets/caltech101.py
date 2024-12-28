import os
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class CalTech101DataModule:

    @staticmethod
    def load_datasets(data_dir: str, image_size: Tuple[int] = (224, 224), seed: int = 2025, use_randaugment: bool = False):
        # 定义图像转换
        if use_randaugment:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        val_test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        class CustomDataset(Dataset):
            def __init__(self, root_dir, transform=None, mode='train'):
                self.root_dir = root_dir
                self.transform = transform
                self.mode = mode
                
                self.image_paths = []
                self.labels = []

                if mode in ['train', 'val', 'test']:
                    self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
                    self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                    for class_name in self.classes:
                        class_dir = os.path.join(root_dir, class_name)
                        for image_name in os.listdir(class_dir):
                            image_path = os.path.join(class_dir, image_name)
                            self.image_paths.append(image_path)
                            self.labels.append(self.class_to_idx[class_name])

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx: int):
                image_path = self.image_paths[idx]
                label = self.labels[idx]
                
                image = Image.open(image_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label

        # 创建训练、验证数据集
        train_dataset = CustomDataset(Path(data_dir) / 'train', transform=train_transform, mode='train')
        val_dataset = CustomDataset(Path(data_dir) / 'val', transform=val_test_transform, mode='val')
        test_dataset = CustomDataset(Path(data_dir) / 'test', transform=val_test_transform, mode='test')

        return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # 测试用的参数，请根据实际情况调整
    image_dir = '/remote-home/pengyichen/diffusion/code/datafiles/processed-caltech-101'  # 替换为你的Caltech-101数据集路径
    batch_size = 4
    num_workers = 2
    image_size = (256, 256)

    # 使用静态方法加载数据集
    train_dataset, val_dataset, test_dataset = CalTech101DataModule.load_datasets(
        data_dir=image_dir, image_size=image_size)

    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 打印数据集大小
    print(f'Training dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    # 测试获取单个样本
    for i in range(3):  # 获取前三个样本进行测试
        image, label = train_dataset[i]
        print(f'Sample {i}: Image shape: {image.shape}, Label: {label}')

    # 测试 DataLoader
    for images, labels in val_loader:
        print(f'Batch image shape: {images.shape}')
        print(f'Batch labels shape: {labels.shape}')
        break
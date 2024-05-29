import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os
import numpy as np
from PIL import Image
import json
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch
class AlbumentationsTransform:
    def __init__(self, apply_augmentation=False):
        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.transforms = [
                A.HorizontalFlip(p=1),
                A.Rotate(limit=35, p=1),
                A.RandomBrightnessContrast(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=1),
                A.RandomGamma(p=1),
                A.GaussNoise(p=1)
            ]
            self.normalize_and_resize = A.Compose([
                A.CLAHE(clip_limit=1.5, tile_grid_size=(3, 3), p=1),
                A.Normalize([0.3421006, 0.3421006, 0.3421006], std = [0.2168359, 0.21683595, 0.21683595]),
                A.Resize(224, 224),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.CLAHE(clip_limit=1.5, tile_grid_size=(3, 3), p=1),
                A.Normalize([0.3421006, 0.3421006, 0.3421006], std = [0.2168359, 0.21683595, 0.21683595]),
                A.Resize(224, 224),
                ToTensorV2()
            ])

    def __call__(self, img, mask, augment_index=None):
        if self.apply_augmentation:
            if augment_index is not None:
                transform = self.transforms[augment_index % len(self.transforms)]
                transform = A.Compose([transform, self.normalize_and_resize])
                augmented = transform(image=img, mask=mask)
                return augmented['image'], augmented['mask']
            else:
                transform = random.choice(self.transforms)
                augmented = transform(image=img, mask=mask)
                augmented = self.normalize_and_resize(image=augmented['image'], mask=augmented['mask'])
                return augmented['image'], augmented['mask']
        else:
            augmented = self.transform(image=img, mask=mask)
            return augmented['image'], augmented['mask']


class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, apply_augmentation=False, augmentation_factor=10):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.apply_augmentation = apply_augmentation
        self.augmentation_factor = augmentation_factor
        self.transform = AlbumentationsTransform(apply_augmentation)
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, idx):
        original_idx = idx // self.augmentation_factor
        augment_idx = idx % self.augmentation_factor

        img_path = os.path.join(self.img_dir, self.img_files[original_idx])
        img = np.array(Image.open(img_path).convert('RGB'))
        annotation_file = os.path.join(self.annotations_dir, os.path.splitext(self.img_files[original_idx])[0] + '.jpg.json')
        with open(annotation_file) as f:
            annotations = json.load(f)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
        for obj in annotations.get('objects', []):
            class_index = int(obj['classTitle']) - 1
            exterior_points = obj['points']['exterior']
            exterior_points = np.array([exterior_points], dtype=np.int32)
            cv2.fillPoly(mask, exterior_points, class_index + 1)

        img, mask = self.transform(img, mask, augment_index=augment_idx)

        # Add debug prints
        #print(f"Image shape after transform: {img.shape}, Mask shape after transform: {mask.shape}")

        return img, mask

    def __len__(self):
        return len(self.img_files) * self.augmentation_factor

def visualize_dataset(dataset, num_samples=5, num_augments=10, num_classes=33):
    to_pil_image = ToPILImage()
    
    for i in range(num_samples):
        fig, axs = plt.subplots(num_augments, 2, figsize=(10, num_augments * 5))
        
        for j in range(num_augments):
            idx = i * num_augments + j
            img, mask = dataset[idx]

            # Normalize image back to [0, 1] range
            img = img * torch.tensor([0.2168359, 0.21683595, 0.21683595]).view(3, 1, 1) + torch.tensor([0.3421006, 0.3421006, 0.3421006]).view(3, 1, 1)
            
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img)
            
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            print(f"Sample {i}, Augment {j}: img shape = {img.size}, mask shape = {mask.shape}")

            if mask.ndim == 2:
                color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                colors = plt.cm.get_cmap('tab10', num_classes)
                
                for c in range(1, num_classes + 1):
                    color_mask[mask == c] = np.array(colors(c)[:3]) * 255
                
                color_mask = Image.fromarray(color_mask)
                
                axs[j, 0].imshow(img)
                axs[j, 0].set_title('Image')
                axs[j, 0].axis('off')
                
                axs[j, 1].imshow(color_mask)
                axs[j, 1].set_title('Mask')
                axs[j, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

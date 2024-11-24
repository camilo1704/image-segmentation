import torch
from PIL import Image
import numpy as np
import torch

class BaseDataset:
    def __init__(self, images_list, transform=None, is_training=True):
        self.transform = transform
        self.is_training = is_training
        self.images = images_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        # Apply transformations if defined
        if self.transform:
            image = self.transform(image, self.is_training)
        return image
        

class SegmentationDataset(BaseDataset):
    def __init__(self, images_list, masks_list, transform=None, is_training=True):
        super().__init__(images_list, transform, is_training)
        self.masks = masks_list

    def __getitem__(self, idx):
        # Load image using the base class
        image = super().__getitem__(idx)
        # Load mask
        mask_path = self.masks[idx]
        mask = np.load(mask_path)
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
        # Apply transformations if defined
        if self.transform:
            image, mask = self.transform(image, mask, self.is_training)
        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask)).float()
        return image, mask

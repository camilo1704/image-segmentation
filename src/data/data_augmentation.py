import random
import torchvision.transforms.functional as TF


class JointTransform:
    def __init__(self, train_image_transform=None, train_mask_transform=None, val_image_transform=None, val_mask_transform=None):
        self.train_image_transform = train_image_transform
        self.train_mask_transform = train_mask_transform
        self.val_image_transform = val_image_transform
        self.val_mask_transform = val_mask_transform

    def __call__(self, image, mask, is_training=True):
        if is_training:
            # Apply shared geometric transformations for training
            if random.random() > 0.5:  # Horizontal flip
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.8:  # Vertical flip
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            if random.random() > 0.5:  # Random rotation
                angle = random.uniform(-30, 30)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Apply image-only transformations for training
            if self.train_image_transform:
                image = self.train_image_transform(image)

            # Apply mask-only transformations for training
            if self.train_mask_transform:
                mask = self.train_mask_transform(mask)
            else:
                mask = TF.to_tensor(mask)
        else:
            # Apply image-only transformations for validation
            if self.val_image_transform:
                image = self.val_image_transform(image)

            # Apply mask-only transformations for validation
            if self.val_mask_transform:
                mask = self.val_mask_transform(mask)
            else:
                mask = TF.to_tensor(mask)

        return image, mask

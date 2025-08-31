import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch


# Albumentations

def get_train_transform():
    return A.Compose([
        A.Transpose(0.5),
        A.GaussNoise(0.3),
        A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
        A.OneOf([
                A.OpticalDistortion(p=0.3),
            ], p=0.2),
        A.OneOf([
                A.RandomBrightnessContrast(),
                ], p=0.3),
        A.HueSaturationValue(p=0.3),

        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

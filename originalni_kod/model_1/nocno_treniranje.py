import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import timm
import pydicom
import os
from PIL import Image
from torchsummary import summary
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR, OneCycleLR

import random
import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import sys
from tqdm.notebook import trange, tqdm
import ipywidgets as widgets
import time
import lmdb
import pickle
from sklearn.metrics import roc_curve, auc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout import CoarseDropout
import traceback
import json



import utils
import treniranja

#print(" Current working directory:", os.getcwd())

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)


#mean, std = utils.mean_std_calc(train_dataset) 
utils.seed_everything(42)
def numpy_to_pil(x):
    return Image.fromarray(x.astype(np.uint8))
transform = transforms.Compose([
    transforms.Lambda(numpy_to_pil),    
    transforms.Resize((224,224), antialias=True),  # Resize using PIL (efficient)
    transforms.ToTensor(),  # Convert to tensor (H, W, C) → (C, H, W)
])
train_mapper, val_mapper, test_mapper, benign_percent  = utils.data_spliter("labels.npy",train_ratio=0.9, val_ratio=0.1
                                                                            , test_ratio=0)
mean = torch.tensor([0.6727, 0.4556, 0.5972]) # ovo za zadani seed
std = torch.tensor([0.1633, 0.0413, 0.0399])


image_size = 224 

train_transforms_simple = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(image_size,image_size),
    A.CoarseDropout(
        max_holes=7,
        min_holes=3,
        max_height=int(image_size * 0.1),
        min_height=int(image_size * 0.1),
        max_width=int(image_size * 0.1),
        min_width=int(image_size * 0.1),
        fill_value=0,
        mask_fill_value=None,p=0.7),
    A.Normalize(),
    ToTensorV2()

])

train_transforms_simple2 = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.Resize(image_size,image_size),
    A.CoarseDropout(
        max_holes=7,
        min_holes=3,
        max_height=int(image_size * 0.1),
        min_height=int(image_size * 0.1),
        max_width=int(image_size * 0.1),
        min_width=int(image_size * 0.1),
        fill_value=0,
        mask_fill_value=None,p=0.7),
    A.Normalize(),
    ToTensorV2()

])



train_transforms = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),

    A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), contrast_limit=(0.0, 0.2), p=0.4),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=(-20, -5), p=0.4),

    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),

    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),

    A.Resize(image_size, image_size),
    A.CoarseDropout(
        max_holes=1,
        min_holes=1,
        max_height=int(image_size * 0.375),
        min_height=int(image_size * 0.375),
        max_width=int(image_size * 0.375),
        min_width=int(image_size * 0.375),
        fill_value=0,
        mask_fill_value=None,
        p=0.7),

    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])


train_transforms2 = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(image_size,image_size),
    A.OneOf([
        A.CoarseDropout(
            max_holes=1,
            min_holes=1,
            max_height=int(image_size * 0.375),
            min_height=int(image_size * 0.375),
            max_width=int(image_size * 0.375),
            min_width=int(image_size * 0.375),
            fill_value=0,
            mask_fill_value=None,p=0.7),
        A.CoarseDropout(
            max_holes=7,
            min_holes=5,
            max_height=int(image_size * 0.15),
            min_height=int(image_size * 0.15),
            max_width=int(image_size * 0.15),
            min_width=int(image_size * 0.15),
            fill_value=0,
            mask_fill_value=None,p=0.7),
        ], p=1),
    A.Normalize(),
    ToTensorV2()

])



test_transforms = A.Compose([
    A.Resize(image_size,image_size),
    A.Normalize(),
    ToTensorV2()
])


train_dataset = utils.LMDBImageDataset("data/data.lmdb","labels.npy", train_mapper, transform=test_transforms)

batch_size = 32

sampler = utils.OversampleMinoritySampler(train_dataset, oversample_factor= 25)




train_dataset1 = utils.LMDBImageDataset_metadata("data/data.lmdb","features.npy", "labels_diagnosis.npy", train_mapper, transform=train_transforms_simple)
train_dataset2 = utils.LMDBImageDataset_metadata("data/data.lmdb","features.npy", "labels_diagnosis.npy", train_mapper, transform=train_transforms_simple2)

val_dataset = utils.LMDBImageDataset_metadata("data/data.lmdb", "features.npy", "labels_diagnosis.npy", val_mapper,transform=test_transforms)




#train_dataset1 = utils.LMDBImageDataset("data/data.lmdb","labels.npy", train_mapper, transform=train_transforms_simple)
#train_dataset2 = utils.LMDBImageDataset("data/data.lmdb", "labels.npy", train_mapper, transform=train_transforms)





#val_dataset = utils.LMDBImageDataset("data/data.lmdb", "labels.npy", val_mapper, transform=test_transforms)



batch_size = 8



'''train_dataset_jpg = utils.LMDBImageDataset("data/data_jpg.lmdb", "labels.npy", train_mapper, transform=train_transforms)
val_dataset_jpg = utils.LMDBImageDataset("data/data_jpg.lmdb", "labels.npy", val_mapper,transform=test_transforms)

batch_size = 8'''


train_loader1 = DataLoader(train_dataset1, batch_size=batch_size,sampler = sampler)
train_loader2 = DataLoader(train_dataset1, batch_size=batch_size,sampler = sampler)

#train_loader1 = DataLoader(train_dataset1, batch_size=batch_size,shuffle = False)
#train_loader2 = DataLoader(train_dataset1, batch_size=batch_size,shuffle = False)


val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# tu cu ubaciti treniranja
train_jobs = [
    #{"name": "convnext_1", "func": treniranja.treniranje_connext_1},
    {"name": "convnext_metadata1", "func": treniranja.treniranje_connext_1_metadata},
    {"name": "convnext_metadata2", "func": treniranja.treniranje_connext_2_metadata},
    #{"name": "convnext_3", "func": treniranja.treniranje_efficient_3},
    #{"name": "convnext_4", "func": treniranja.treniranje_efficient_4},
]

# Your loaders (you can pass different loaders if needed)

num_epochs = 15
for job in train_jobs:
    for i,loader in enumerate([train_loader1,train_loader2]):
        print(f"\n Starting training: {job['name']}, num_epochs: {num_epochs}")
        try:
            # Call the training function
            model, best_model,logs = job["func"](loader, val_loader, num_epochs)
            torch.save(model.state_dict(), f"modeli/{job['name']}_loader_{i}")
            torch.save(best_model, f"modeli/best_{job['name']}_loader_{i}")
            print(f" Training completed: {job['name']}")
            log_path = f"modeli/{job['name']}_loader_{i}.json"
            print(f" Final AUC: {logs[f'epoch_{num_epochs}']['auc']:.3f}")
            print(f" Final loss: {logs[f'epoch_{num_epochs}']['loss']:.4f}")
            print(f"  Total time: {sum(log['time'] for log in logs.values()):.1f} seconds")
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
    
        except Exception as e:
            print(f" Error during training {job['name']}: {e}")
            traceback.print_exc()
            continue  # Skip to next training function

print(" All training jobs completed.")

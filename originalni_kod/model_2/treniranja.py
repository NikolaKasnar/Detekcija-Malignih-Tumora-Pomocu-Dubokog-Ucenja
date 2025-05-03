import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import timm
import pydicom
import os
from PIL import Image
from torchsummary import summary
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR, OneCycleLR,LambdaLR

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
import time
from contextlib import suppress
#from torch.amp import autocast

import utils

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvNeXtBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base_model = models.convnext_small(pretrained=True)
        
        # Backbone: features + avgpool + flatten
        self.backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,  # output: [B, 768, 1, 1]
            nn.Flatten()         # → [B, 768]
        )

        # Custom fully connected head
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary output
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

class ImprovedConvNeXtBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.5, use_attention=True):
        super().__init__()
        base_model = models.convnext_small(pretrained=True)
        
        # Backbone
        self.backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten()
        )
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(768, 768//4),
                Swish(),
                nn.Linear(768//4, 768),
                nn.Sigmoid()
            )
        
        # Enhanced head
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Dropout(dropout*0.8),  # Slightly less dropout in deeper layers
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
            
        return self.head(features)

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

def constant_lr_scheduler(optimizer, group_idx, constant_lr):
    def lr_lambda(epoch):
        return constant_lr / optimizer.param_groups[group_idx]['initial_lr']
    return LambdaLR(optimizer, lr_lambda)


def log_epoch_stats(log_dict, epoch, loss_value, epoch_time,auc):
    log_dict[f"epoch_{epoch+1}"] = {
        "time": round(epoch_time, 2),
        "loss": round(loss_value, 5),
        "auc": round(auc,3)
    }

def train_model(
    model, train_loader,val_loader, optimizer, 
    scheduler, fine_tune_lr,
    criterion, device, fine_tune_lr_backbone = 1e-5,
    num_epochs=20, freeze_epochs=2, finetune_head_epoch=10,
    
):
    scaler = GradScaler()
    logs = {}

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone(model)

        if epoch == finetune_head_epoch:
            print(" Switching head to constant finetune LR")
            scheduler = None
            optimizer.param_groups[0]['lr'] = fine_tune_lr_backbone
            optimizer.param_groups[1]['lr'] = fine_tune_lr

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_index, (inputs, labels) in enumerate(train_loader):
            time.sleep(0.05)  # slow down to reduce heat

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if scheduler:
                scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        tpr, fpr, auc = utils.draw_roc(model, val_loader, draw = False)
        #  Log this epoch
        log_epoch_stats(logs, epoch, avg_loss, epoch_time, auc)
        print(f"   current AUC: {auc}")
    return model, logs

def train_model_2(
    model, train_loader, val_loader, optimizer, 
    scheduler, fine_tune_lr,
    criterion, device, fine_tune_lr_backbone = 1e-5,
    num_epochs=20, freeze_epochs=2, finetune_head_epoch=10,
):
    scaler = GradScaler()
    logs = {}

    best_auc = 0
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        # Za prvih par epoha je zamrznut backbone
        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone(model)

        if epoch == finetune_head_epoch:
            print(" Switching head to constant finetune LR")
            scheduler = None
            optimizer.param_groups[0]['lr'] = fine_tune_lr_backbone
            optimizer.param_groups[1]['lr'] = fine_tune_lr

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # Validation AUC
        tpr, fpr, val_auc = utils.draw_roc(model, val_loader, draw=False)
        # Smanjimo LR ako se AUC smanjuje
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_auc)

        # Log
        log_epoch_stats(logs, epoch, avg_loss, epoch_time, val_auc)
        print(f"   Loss: {avg_loss:.4f} | AUC: {val_auc:.4f}")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model_auc.pth")
    
    return model, logs



def improved_train_model(
    model, train_loader, val_loader, optimizer, 
    scheduler, criterion, device, 
    num_epochs=25, freeze_epochs=3, 
    early_stop_patience=5, use_amp=True
):
    scaler = GradScaler(enabled=use_amp)
    logs = {}
    best_auc = 0
    epochs_no_improve = 0
    
    amp_context = autocast(device_type='cuda') if use_amp else suppress()

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        
        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone(model)
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            with amp_context:
                outputs = model(inputs)
                
                # Optional label smoothing
                smooth_labels = labels.float() * 0.9 + 0.05
                loss = criterion(outputs.squeeze(), smooth_labels)

            scaler.scale(loss).backward()

            # Optional gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        tpr, fpr, val_auc = utils.draw_roc(model, val_loader, draw=False)

        # Update scheduler properly
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_patience:
                print(f" Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load('best_model.pth'))
                break
        
        log_epoch_stats(logs, epoch, avg_loss, epoch_time, val_auc)
        print(f"   Loss: {avg_loss:.4f} | AUC: {val_auc:.4f} | Best AUC: {best_auc:.4f}")
    
    return model, logs




#######################################################################################################3
def treniranje_connext_1(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = ConvNeXtBinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    # Tu odabiremo scheduler
    # Smanji LR ako se AUC smanji u par epoha
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model, logs = train_model_2(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-4
        ,criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=2,
        finetune_head_epoch=14
    )
    return model, logs

def treniranje_connext_2(train_loader, val_loader, num_epochs=14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ConvNeXtBinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    model, logs = improved_train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=2,
        early_stop_patience=5,
        use_amp=True
    )

    return model, logs

#########################################################################################################################


class EfficientNetB6BinaryClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base_model = models.efficientnet_b6(pretrained=True)

        # Extract feature extractor (everything except classifier)
        self.backbone = base_model.features

        # EfficientNet-B6 outputs feature maps of shape [B, 2304, 7, 7]
        self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, 2304, 1, 1]
        self.flatten = nn.Flatten()          # → [B, 2304]

        # Custom head with Swish activation (SiLU in PyTorch)
        self.head = nn.Sequential(
            nn.Linear(2304, 256),
            nn.SiLU(),                       # Swish = SiLU
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)                 # Binary output (logit)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


def treniranje_efficient_1(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = EfficientNetB6BinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    scheduler = CyclicLR(
        optimizer,
        base_lr=[1e-5,1e-4],
        max_lr=[1e-4,1e-2],
        step_size_up=len(train_loader) ,
        step_size_down=len(train_loader) ,
        mode='triangular2',
        cycle_momentum=False
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model, logs = train_model(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-4
        ,criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=2,
        finetune_head_epoch=14
    )
    return model, logs

def treniranje_efficient_2(train_loader,val_loader,num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = EfficientNetB6BinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-4,1e-2],
        div_factor = 50,
        final_div_factor =10,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model, logs = train_model(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-5,
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=20,
        freeze_epochs=2,
        finetune_head_epoch=14
    )
    return  model, logs

def treniranje_efficient_3(train_loader,val_loader,num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = EfficientNetB6BinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[1e-3,1e-2],
        div_factor = 50,
        final_div_factor =10,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        cycle_momentum=False
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model, logs = train_model(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-5,
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=20,
        freeze_epochs=12,
        finetune_head_epoch=20
    )
    return  model, logs

def treniranje_efficient_4(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = EfficientNetB6BinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 5e-4, 'initial_lr': 5e-4}
    ], weight_decay=0.01)

    scheduler = CyclicLR(
        optimizer,
        base_lr=[1e-5,5e-5],
        max_lr=[1e-5,2e-2],
        step_size_up=len(train_loader) ,
        step_size_down=len(train_loader) ,
        mode='triangular2',
        cycle_momentum=False
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model, logs = train_model(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-4
        ,criterion=torch.nn.BCEWithLogitsLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=8,
        finetune_head_epoch=14
    )
    return model, logs
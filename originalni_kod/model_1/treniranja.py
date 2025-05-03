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

import utils

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

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

class ConvNeXtBinaryClassifier_metadata(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        base_model = models.convnext_small(pretrained=True)

        # Backbone for image input
        self.backbone1 = nn.Sequential(
            base_model.features,
            base_model.avgpool,  # output: [B, 768, 1, 1]
            nn.Flatten()         # → [B, 768]
        )

        # Backbone for tabular input
        self.backbone2 = nn.Sequential(
            nn.Linear(7, 64),
            Swish(),
            nn.LayerNorm(64),
            #nn.Dropout(dropout),
            nn.Linear(64, 128),
            Swish(),
            nn.LayerNorm(128),
            #nn.Dropout(dropout),
            nn.Linear(128, 32)
        )

        self.sex_embed = nn.Embedding(num_embeddings=2, embedding_dim=2)
        self.site_embed = nn.Embedding(num_embeddings=7, embedding_dim=4)

        self.backbone2.apply(init_weights)

        # Combined head
        self.head = nn.Sequential(
            nn.Linear(768 + 32, 256),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(64, 9)  # or 1 if binary
        )

    def forward(self, x1, x2):
        sex_idx = x2[:, 0].long()
        age_scalar = x2[:, 1]         

        site_onehot = x2[:, 2:]          
        site_idx = site_onehot.argmax(dim=1)  
    
        sex_emb = self.sex_embed(sex_idx)             # [B, 2]
        site_emb = self.site_embed(site_idx)          # [B, 4]
        age_scalar = age_scalar.unsqueeze(1)          # [B, 1]

        tabular_input = torch.cat([sex_emb, age_scalar, site_emb], dim=1)  # [B, 7]

        x1 = self.backbone1(x1)
        x2 = self.backbone2(tabular_input)
        x3 = torch.cat([x1, x2], dim=1)
        return self.head(x3)


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True

def freeze_backbone_metadata(model):
    for param in model.backbone1.parameters():
        param.requires_grad = False

def unfreeze_backbone_metadata(model):
    for param in model.backbone1.parameters():
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
    
    best_auc = 0


    scaler = GradScaler()
    logs = {}
    scheduler_in_use = None
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone(model)

        if epoch == finetune_head_epoch:
            scheduler_in_use = scheduler

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

            if scheduler_in_use:
                scheduler_in_use.step()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        tpr, fpr, auc = utils.draw_roc(model, val_loader, draw = False)
        #  Log this epoch
        log_epoch_stats(logs, epoch, avg_loss, epoch_time, auc)
        print(f"   current AUC: {auc}")

        if auc > best_auc:
                best_auc = auc
                best_model = model.state_dict()
    
    return model,best_model, logs

def train_model_2(
    model, train_loader, val_loader, optimizer, 
    scheduler, fine_tune_lr,
    criterion, device, fine_tune_lr_backbone = 1e-5,
    num_epochs=20, freeze_epochs=2, finetune_head_epoch=10,
):
    scaler = GradScaler()
    logs = {}

    best_model = None

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
                #print(outputs.shape)
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
            best_model = model.state_dict()
    
    return model,best_model, logs

def train_model_metadata(
    model, train_loader,val_loader, optimizer, 
    scheduler, fine_tune_lr,
    criterion, device, fine_tune_lr_backbone = 1e-5,
    num_epochs=20, freeze_epochs=2, finetune_head_epoch=10,
    
):
    
    best_auc = 0
    scaler = GradScaler()
    logs = {}
    scheduler_in_use = None
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")

        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone_metadata(model)

        if epoch == finetune_head_epoch:
            scheduler_in_use = scheduler

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_index, (inputs,features, labels) in enumerate(train_loader):
            time.sleep(0.05)  # slow down to reduce heat

            inputs, features,labels = inputs.to(device),features.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs,features)
                labels_idx = labels.argmax(dim=1) if labels.ndim == 2 else labels  # handle one-hot or already-indexed
                loss = criterion(outputs, labels_idx)

            scaler.scale(loss).backward()
            for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"❗ NaNs in gradient of {name}")


            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if scheduler_in_use:
                if isinstance(scheduler_in_use, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_in_use.step(auc)
                else:
                    scheduler_in_use.step()


        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        auc = utils.compute_global_auc(model, val_loader)
        #  Log this epoch
        log_epoch_stats(logs, epoch, avg_loss, epoch_time, auc)
        print(f"   current AUC: {auc}")

        
        if auc > best_auc:
            best_auc = auc
            best_model = model.state_dict()
    
    return model,best_model, logs



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
    
    # Mixed precision training context manager
    amp_context = autocast if use_amp else suppress

    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        
        # Unfreeze logic
        if epoch == freeze_epochs:
            print(" Unfreezing backbone!")
            unfreeze_backbone(model)
            # Reduce LR after unfreezing
            for g in optimizer.param_groups:
                g['lr'] *= 0.1

        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            with amp_context():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                
                # Label smoothing
                smooth_labels = labels.float() * 0.9 + 0.05
                loss = criterion(outputs.squeeze(), smooth_labels)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        tpr, fpr, val_auc = utils.draw_roc(model, val_loader, draw=False)
        
        # Early stopping check
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

def treniranje_connext_2(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = ConvNeXtBinaryClassifier(dropout=0.5).to(device)
    freeze_backbone(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-6, 'initial_lr': 1e-6},
        {'params': model.head.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5}
    ], weight_decay=0.01)

    # Tu odabiremo scheduler
    # Smanji LR ako se AUC smanji u par epoha
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4],
        div_factor= 50,
        final_div_factor = 2, 
        steps_per_epoch=len(train_loader),  # adjust based on your dataloader
        epochs=14
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
        freeze_epochs=3,
        finetune_head_epoch=6,
    )
    return model, logs

def treniranje_connext_1_metadata(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = ConvNeXtBinaryClassifier_metadata(dropout=0.3).to(device)
    freeze_backbone_metadata(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone1.parameters(), 'lr': 5e-7, 'initial_lr': 5e-7},
        {'params': model.backbone2.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.sex_embed.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.site_embed.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5}
    ], weight_decay=0.01)
    # Tu odabiremo scheduler
    # Smanji LR ako se AUC smanji u par epoha
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model,best_model, logs = train_model_metadata(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-4
        ,criterion= torch.nn.CrossEntropyLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=2,
        finetune_head_epoch=4,
    )
    return model,best_model, logs

def treniranje_connext_2_metadata(train_loader,val_loader, num_epochs = 14):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    model = ConvNeXtBinaryClassifier_metadata(dropout=0.3).to(device)
    freeze_backbone_metadata(model)

    optimizer = torch.optim.AdamW([
        {'params': model.backbone1.parameters(), 'lr': 2e-6, 'initial_lr': 2e-6},
        {'params': model.backbone2.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-5, 'initial_lr': 1e-5}
    ], weight_decay=0.01)

    # Tu odabiremo scheduler
    # Smanji LR ako se AUC smanji u par epoha
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[5e-5, 5e-4,5e-4],
        div_factor= 100,
        final_div_factor = 1/10, 
        steps_per_epoch=len(train_loader),  # adjust based on your dataloader
        epochs=9
    )

    #  Constant LR scheduler for head after epoch 10
    scheduler_head_finetune = constant_lr_scheduler(optimizer, group_idx=1, constant_lr=5e-4)

    #  Train
    model,best_model, logs = train_model_metadata(
        model, train_loader,val_loader, optimizer,
        scheduler, 1e-4
        ,criterion= torch.nn.CrossEntropyLoss(),
        device=device,
        num_epochs=num_epochs,
        freeze_epochs=2,
        finetune_head_epoch=4,
    )
    return model,best_model, logs



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
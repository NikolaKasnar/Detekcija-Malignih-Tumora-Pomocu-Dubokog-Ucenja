import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import timm
import pydicom
import os
from PIL import Image
from torchsummary import summary
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
import io
from torchvision.ops import sigmoid_focal_loss


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
from sklearn.metrics import roc_curve, auc,roc_auc_score

from sklearn.metrics import roc_curve, auc

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)

##################################################################################################### spremanje podataka

def LMDBsetup(data_folder, output_lmdb, label_path, label_output_path, target_size = (384, 384)):
    df = pd.read_csv(label_path ) # load data labels
    df.sort_values(by='image_name', ascending=True, inplace=True)
    np.save(label_output_path, df["target"].to_numpy())

    # --- INIT LMDB ---
    file_names = sorted(os.listdir(data_folder))
    map_size = 50 * 1024 * 1024 * 1024  # 50 GB
    env = lmdb.open(output_lmdb, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, file_name in enumerate(file_names):
            if not file_name.endswith(".dcm"):
                continue

            path = os.path.join(data_folder, file_name)
            ds = pydicom.dcmread(path)
            array = ds.pixel_array  # shape: (H, W) or (H, W, 3)

            # Ensure 3 channels
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)

            # Convert to PIL and resize to (640, 480)
            image = Image.fromarray(array.astype(np.uint8))
            image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
            resized_array = np.array(image_resized)  # shape: (480, 640, 3)

            # Serialize as np.uint8 array
            image_bytes = pickle.dumps(resized_array)

            # Key like "image_000001"
            key = f"image_{idx:06d}".encode("ascii")
            txn.put(key, image_bytes)

        # Store count as metadata
        txn.put(b"__len__", str(len(file_names)).encode("ascii"))

    env.close()
    print(f" Done. Saved {len(file_names)} resized images to {output_lmdb}")
    return


###############################################################################################3 funkcije i klase za učitavanje podataka

class LMDBImageDataset(Dataset):
    def __init__(self, lmdb_path,  label_path,data_mapper, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.labels = np.load(label_path)
        self.env = None  # LMDB env will be opened lazily per worker
        self.data_mapper = data_mapper # ovo koristim kako bih mogao raditi train/test splitove samo kao liste pa ucitavati sve iz istog foldera

        # Read the number of samples (stored during creation)
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                length_bytes = txn.get(b"__len__")
                if length_bytes is None:
                    raise ValueError("Could not find '__len__' key in LMDB!")
                self.length = len(self.data_mapper)
        print(f"Dataset initialized. Total length: {self.length}")

    def __getitem__(self, idx):
        # Open LMDB in each worker process
        idx = self.data_mapper[idx]
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        with self.env.begin(write=False) as txn:
            key = f"image_{idx:06d}".encode("ascii")
            image_bytes = txn.get(key)
            image = pickle.loads(image_bytes)  # shape: (480, 640, 3), dtype: uint8
        # Optionally apply transform
        if self.transform:
            image = self.transform(image=image)["image"]  #ovo ako koristimo albumations transforms 
            #image = self.transform(image)
            

        label = self.labels[idx]
        
        return image,label

    def __len__(self):
        return self.length
    
class LMDBImageDataset_metadata(Dataset):
    def __init__(self, lmdb_path,features_path,  label_path,data_mapper, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.features = np.load(features_path, allow_pickle=True)
        self.labels = np.load(label_path, allow_pickle=True)
        self.env = None  # LMDB env will be opened lazily per worker
        self.data_mapper = data_mapper # ovo koristim kako bih mogao raditi train/test splitove samo kao liste pa ucitavati sve iz istog foldera

        # Read the number of samples (stored during creation)
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                length_bytes = txn.get(b"__len__")
                if length_bytes is None:
                    raise ValueError("Could not find '__len__' key in LMDB!")
                self.length = len(self.data_mapper)
        print(f"Dataset initialized. Total length: {self.length}")

    def __getitem__(self, idx):
        # Open LMDB in each worker process
        idx = self.data_mapper[idx]
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        with self.env.begin(write=False) as txn:
            key = f"image_{idx:06d}".encode("ascii")
            image_bytes = txn.get(key)
            image = pickle.loads(image_bytes)  # shape: (480, 640, 3), dtype: uint8
        # Optionally apply transform
        if self.transform:
            image = self.transform(image=image)["image"]  #ovo ako koristimo albumations transforms 
            #image = self.transform(image)
            
        features = torch.tensor(self.features[idx,:], dtype=torch.float32)
        sex_value = self.features[idx, 0]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image,features,label

    def __len__(self):
        return self.length
    
class LMDBImageJPGDataset(Dataset):
    def __init__(self, lmdb_path, label_path, data_mapper, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.labels = np.load(label_path)
        self.env = None  # LMDB env will be opened lazily per worker
        self.data_mapper = data_mapper  # maps dataset indices to LMDB keys

        # Read total number of mapped items (from data_mapper)
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                if txn.get(b"__len__") is None:
                    raise ValueError("Could not find '__len__' key in LMDB!")
                self.length = len(self.data_mapper)

        print(f"JPEG LMDB Dataset initialized. Total length: {self.length}")

    def __getitem__(self, idx):
        mapped_idx = self.data_mapper[idx]

        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        with self.env.begin(write=False) as txn:
            key = f"image_{mapped_idx:06d}".encode("ascii")
            jpeg_bytes = txn.get(key)
            if jpeg_bytes is None:
                raise IndexError(f"No image found at key: image_{mapped_idx:06d}")

            # Decode JPEG into PIL
            image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image=image)["image"]

        label = self.labels[mapped_idx]
        return image, label

    def __len__(self):
        return self.length
    
class LMDBImageJPGDataset_metadata(Dataset):
    def __init__(self, lmdb_path,features_path,  label_path,data_mapper, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.features = np.load(features_path)
        self.labels = np.load(label_path)
        self.env = None  # LMDB env will be opened lazily per worker
        self.data_mapper = data_mapper  # maps dataset indices to LMDB keys

        # Read total number of mapped items (from data_mapper)
        with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                if txn.get(b"__len__") is None:
                    raise ValueError("Could not find '__len__' key in LMDB!")
                self.length = len(self.data_mapper)

        print(f"JPEG LMDB Dataset initialized. Total length: {self.length}")

    def __getitem__(self, idx):
        mapped_idx = self.data_mapper[idx]

        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

        with self.env.begin(write=False) as txn:
            key = f"image_{mapped_idx:06d}".encode("ascii")
            jpeg_bytes = txn.get(key)
            if jpeg_bytes is None:
                raise IndexError(f"No image found at key: image_{mapped_idx:06d}")

            # Decode JPEG into PIL
            image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image=image)["image"]
        
        
        features = self.features[mapped_idx,:]
        label = self.labels[mapped_idx, :]
        
        return image,features,label

    def __len__(self):
        return self.length
    
def get_class_indices(dataset):
    class_to_indices = {}
    class_label_fn = lambda x: x[1]
    for idx in range(len(dataset)):
        label = int(class_label_fn(dataset[idx]))  # this works now
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    return class_to_indices
    
class OversampleMinoritySampler(Sampler):
    def __init__(self, dataset, oversample_factor=25):
        self.class_to_indices = get_class_indices(dataset)
        self.oversample_factor = oversample_factor

        self.minority_class = 1
        self.majority_class = 0
        self.indices = self.class_to_indices[self.majority_class] + \
              self.class_to_indices[self.minority_class] * self.oversample_factor
        

    def __iter__(self):
        indices = self.class_to_indices[self.majority_class] + \
              self.class_to_indices[self.minority_class] * self.oversample_factor
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.indices)


def split_indices(indices, ratios):
    n = len(indices)
    a = int(ratios[0] * n)
    b = int(ratios[1] * n)
    return indices[:a], indices[a:a+b], indices[a+b:]

def data_spliter(label_path, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1):
    labels = np.load(label_path)
    n = len(labels)

    idx_1 = np.where(labels == 1)[0]
    idx_0 = np.where(labels == 0)[0]

    np.random.shuffle(idx_1)
    np.random.shuffle(idx_0)

    train_1, val_1, test_1 = split_indices(idx_1, (train_ratio, val_ratio, test_ratio))
    train_0, val_0, test_0 = split_indices(idx_0, (train_ratio, val_ratio, test_ratio))

    # Step 4: Combine splits
    train_idx = np.concatenate([train_1, train_0])
    val_idx   = np.concatenate([val_1, val_0])
    test_idx  = np.concatenate([test_1, test_0])

    # Step 5: Sort indices (optional)
    train_idx = np.sort(train_idx)
    val_idx   = np.sort(val_idx)
    test_idx  = np.sort(test_idx)
    benign_malignant_ratio = len(idx_0)/n

    return train_idx, val_idx, test_idx, benign_malignant_ratio


def numpy_to_pil(x):
    return Image.fromarray(x.astype(np.uint8))

def mean_std_calc(dataset):    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0
    br = 0
    for batch_i, (images, _) in enumerate(tqdm(dataloader)):
        #images = images.float() / 255.0 
        batch_samples = images.size(0)  # Batch size
        #print(images.shape)
        #images = images.view(batch_samples, -1)   
        mean += images.mean(dim=[0,2,3]) * batch_samples
        std += images.std(dim=[0,2,3]) * batch_samples
        num_samples += batch_samples

    # Final mean and std per channel
    mean /= num_samples #tensor([136.4660])
    std /= num_samples  #tensor([14.7633])

    return mean, std



def seed_everything(seed=42):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # current GPU
    torch.cuda.manual_seed_all(seed)  # all GPUs

    # Ensure deterministic behavior (can slow down training!)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f" Seed set to: {seed}")


############################################################################################3 klase za treniranje


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()  # AMP
        torch.save(model.state_dict(), 'init_params.pt')
    
    def range_test(self, iterator, end_lr=10, num_iter=100,
               smooth_f=0.05, diverge_th=5, just_head=0):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter, just_head)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            lrs.append(lr_scheduler.get_last_lr()[0])

            # update lr
            lr_scheduler.step()

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        self.model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = iterator.get_batch()
        x = x.to(self.device)
        y = y.to(self.device)

        with autocast():  # Mixed precision context
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()



class ExponentialLR:
    def __init__(self, optimizer, end_lr, num_iter, group_idx=0):  
        self.optimizer = optimizer
        self.group_idx = group_idx
        self.init_lr = optimizer.param_groups[group_idx]['lr']
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.step_num = 0
        self.mult = (end_lr / self.init_lr) ** (1 / num_iter)

    def step(self):
        self.step_num += 1
        # Only update head's param group
        self.optimizer.param_groups[self.group_idx]['lr'] *= self.mult

    def get_last_lr(self):
        return [self.optimizer.param_groups[self.group_idx]['lr']]



class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss
        
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, alpha=0.85, reduction='mean'):
        """
        gamma_pos: focusing parameter for positive samples
        gamma_neg: focusing parameter for negative samples
        alpha: weighting factor for positive class (like in classic focal loss)
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits, shape [B]
        targets: binary labels (0 or 1), shape [B]
        """
        if inputs.ndim > 1:
            inputs = inputs.view(-1)
        if targets.ndim > 1:
            targets = targets.view(-1)

        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Positives: targets == 1
        pos_term = -self.alpha * (1 - probs) ** self.gamma_pos * torch.log(probs + 1e-8)
        # Negatives: targets == 0
        neg_term = -(1 - self.alpha) * (probs) ** self.gamma_neg * torch.log(1 - probs + 1e-8)

        loss = targets * pos_term + (1 - targets) * neg_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



class TorchVisionFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs, targets.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )
    
###########################################################################################3 funkcije za treniranje

def lr_finder(model, train_loader, optimizer, criterion =nn.CrossEntropyLoss(), just_head = 0, END_LR = 100, NUM_ITER = 100):
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    lr_finder = LRFinder(model, optimizer, criterion, device)
    group_idx = 1 if just_head else 0
    lrs, losses = lr_finder.range_test(train_loader, END_LR, NUM_ITER, just_head=group_idx)
    
    skip_start = 5
    skip_end = 5
    lrs = lrs[skip_start:-skip_end]
    losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()


class CustomCyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, 
                 step_size_up, step_size_down=None,
                 cycle_multipliers=None,
                 mode='triangular2', last_epoch=-1):

        self.base_lrs = base_lr if isinstance(base_lr, list) else [base_lr] * len(optimizer.param_groups)
        self.max_lrs = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.cycle_multipliers = cycle_multipliers or [1.0, 1.5, 2.0, 3.5]  # Can be infinite generator

        self.total_steps = self.step_size_up + self.step_size_down
        self.cycle_count = 0
        self.step_in_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []

        # Calculate current cycle position
        cycle_progress = self.step_in_cycle / self.total_steps
        scale = 1.0
        if self.mode == 'triangular2':
            scale = 1.0 / (2.0 ** self.cycle_count)

        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            if self.step_in_cycle <= self.step_size_up:
                pct = self.step_in_cycle / self.step_size_up
            else:
                pct = 1 - ((self.step_in_cycle - self.step_size_up) / self.step_size_down)

            lr = base_lr + (max_lr - base_lr) * pct * scale
            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        self.step_in_cycle += 1

        # Check if current cycle is done
        if self.step_in_cycle >= self.total_steps:
            self.step_in_cycle = 0
            self.cycle_count += 1

            # Expand step sizes if defined
            if self.cycle_count < len(self.cycle_multipliers):
                mult = self.cycle_multipliers[self.cycle_count]
                self.step_size_up = int(self.step_size_up * mult)
                self.step_size_down = int(self.step_size_down * mult)
                self.total_steps = self.step_size_up + self.step_size_down

        super().step(epoch)

'''
example usage
scheduler = CustomCyclicLR(
    optimizer,
    base_lr=[1e-5, 1e-4],
    max_lr=[1e-4, 1e-2],
    step_size_up=len(train_loader),
    step_size_down=len(train_loader),
    cycle_multipliers=[1.0, 1.5, 2.0, 2.5,4],  # cycles get longer
    mode='triangular2'
)
'''




#################################################################################################### analiza modela

def draw_roc(model, test_loader, draw = True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1))  # binary case
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1]  # class 1 prob in multiclass

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    if(draw):
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    return tpr, fpr, roc_auc


def compute_global_auc(model, dataloader, num_classes=9, device=None):
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_labels = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # image + metadata
                images, features, labels = batch
                images = images.to(device)
                features = features.to(device)
                outputs = model(images, features)
            else:  # just images
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Stack predictions and labels
    all_logits = np.concatenate(all_logits, axis=0) 
    all_probs = np.concatenate(all_probs, axis=0)  # shape [N, 9]
    malignant_probs = all_probs[:,4]
    all_labels = np.concatenate(all_labels, axis=0)  # shape [N] or [N, 9]

    # Convert labels to one-hot if they're class indices
    if all_labels.ndim == 1:
        binary_labels = (all_labels == 4).astype(np.uint8)
        one_hot_labels = np.eye(num_classes)[all_labels]
    else:
        one_hot_labels = all_labels
        binary_labels = all_labels[:,4]

    # Remove NaNs (e.g. if softmax produced them somehow)
    mask = ~np.isnan(all_probs).any(axis=1) & ~np.isnan(one_hot_labels).any(axis=1)
    if np.sum(mask) == 0:
        print(" All predictions contain NaNs, cannot compute AUC.")
        return None

    all_probs_clean = all_probs[mask]
    one_hot_labels_clean = one_hot_labels[mask]
                                                                                    ################3dijagnostika
    print("\n Class distribution BEFORE removing NaNs:")
    counts_before = np.sum(one_hot_labels, axis=0)
    for i, count in enumerate(counts_before):
        print(f"  Class {i}: {int(count)} samples")

    print("\n Class distribution AFTER removing NaNs:")
    counts_after = np.sum(one_hot_labels_clean, axis=0)
    for i, count in enumerate(counts_after):
        print(f"  Class {i}: {int(count)} samples")
    
    if np.sum(~mask) > 0:
        print(f"\n Found {np.sum(~mask)} samples with NaNs — showing their logits:")
        nan_indices = np.where(~mask)[0]
        for idx in nan_indices:
            print(f"Sample {idx}: logits = {all_logits[idx]}")


    # 4. Print warning if any classes were fully dropped
    print("\n Classes dropped due to NaNs:")
    for i in range(num_classes):
        if counts_before[i] > 0 and counts_after[i] == 0:
            print(f"  Class {i}: {int(counts_before[i])} → 0")
                                                                                ##################################

    # Check if there are at least two classes present
    n_classes_present = np.unique(np.argmax(one_hot_labels_clean, axis=1)).shape[0]
    if n_classes_present < 2:
        print(" Only one class present in y_true — AUC undefined.")
        return None

    # Compute micro-average AUC
    try:
        fpr,tpr,thresholds = roc_curve(binary_labels,malignant_probs)
        roc_auc = auc(fpr,tpr)
        return roc_auc
    except ValueError as e:
        print(f" AUC computation failed: {e}")
        return None



def probabilities_range(model, test_loader):
    model.eval()
    all_probs = []
    all_labels = []
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Prob range:", min(all_probs), max(all_probs))
    print("Pred class 1 count:", np.sum(np.array(all_probs) > 0.5))
    print("True class 1 count:", np.sum(np.array(all_labels)))


mean = (0.6727, 0.4556, 0.5972) # ovo za zadani seed
std = (0.1633, 0.0413, 0.0399)

def numpy_to_pil(x):
    return Image.fromarray(x.astype(np.uint8))


def build_tta_transforms(mean, std):
    base = [
        T.Lambda(numpy_to_pil),
        T.Resize((224, 224), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]
    
    return [
        T.Compose(base),
        T.Compose([T.Lambda(numpy_to_pil), T.Resize((224, 224), antialias=True), T.RandomHorizontalFlip(p=1.0), *base[2:]]),
        T.Compose([T.Lambda(numpy_to_pil), T.Resize((224, 224), antialias=True), T.RandomVerticalFlip(p=1.0), *base[2:]]),
        T.Compose([T.Lambda(numpy_to_pil), T.Resize((224, 224), antialias=True), T.RandomRotation(degrees=15), *base[2:]])
    ]


def tta_predict(model, image,tta_transforms, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for transform in tta_transforms:
            aug_img = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
            output = model(aug_img)
            prob = torch.sigmoid(output).item()  # Binary prob
            preds.append(prob)
    return sum(preds) / len(preds)  # Average over TTA runs

# Evaluate AUC with TTA over a dataset
from sklearn.metrics import roc_auc_score
def tta_evaluate(model, dataset, device, mean=(0.6727, 0.4556, 0.5972), std= (0.1633, 0.0413, 0.0399)):
    print(" Running TTA Evaluation...")
    tta_transforms = build_tta_transforms(mean, std)

    y_true = []
    y_pred = []

    for i in range(len(dataset)):
        img, label = dataset[i]
        pred = tta_predict(model, img, tta_transforms, device)
        y_pred.append(pred)
        y_true.append(label)

    auc = roc_auc_score(y_true, y_pred)
    print(f" TTA AUC: {auc:.4f}")
    return auc
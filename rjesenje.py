import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
import pandas as pd 
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import argparse
import sys 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

MEDIAN_AGE = 50

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

def preprocess_single_inference_data(sex, age_string, anatomical_site):
    if sex is None or pd.isna(sex): # Handle missing sex
        sex_encoded = 0 # Corresponds to 'female' or missing in training fillna(0)
    elif str(sex).lower() == 'female' or str(sex).lower() == 'f':
        sex_encoded = 0
    elif str(sex).lower() == 'male' or str(sex).lower() == 'm':
        sex_encoded = 1
    else:
        #print(f"Warning: Unrecognized sex value '{sex}'. Defaulting to 0 (female/missing).")
        sex_encoded = 0
    sex_feature = np.float32(sex_encoded) # Ensure float32

    # 2. Process Age
    try:
        
        if age_string and isinstance(age_string, str) and len(age_string) > 0:
            age = int(age_string[:-1])

    except ValueError:
        #print(f"Warning: Could not convert age substring '{age_string[:-1]}' to integer. Setting age to None.")
        age = None 

    except Exception as e:
        #print(f"Warning: Unexpected error processing PatientAge ('{age_string}'): {e}. Setting age to None.")
        age = None 

    if age is None or pd.isna(age): # Handle missing age
        age_processed = MEDIAN_AGE # Use pre-calculated median from training
    else:
        age_processed = np.float32(age) 

    if anatomical_site is None or pd.isna(anatomical_site) or not isinstance(anatomical_site, str) or anatomical_site.strip() == '':
        site_to_encode = 'unknown' # Corresponds to fillna('unknown') in training
    else:
        site_to_encode = anatomical_site.lower().strip() # Normalize input

    # Create the one-hot encoded vector based on training columns
    site_onehot_vector = np.zeros(len(ANATOM_SITE_COLUMNS_TRAIN), dtype=np.float32)

    target_column_name = f'site_{site_to_encode}'

    if target_column_name in ANATOM_SITE_COLUMNS_TRAIN:
        try:
            site_index = ANATOM_SITE_COLUMNS_TRAIN.index(target_column_name)
            site_onehot_vector[site_index] = 1.0 
        except ValueError:
             target_column_name = f'site_{site_to_encode}'
             # Try setting 'unknown' index
             try:
                 site_index = ANATOM_SITE_COLUMNS_TRAIN.index(target_column_name)
                 site_onehot_vector[site_index] = 1.0
             except ValueError:
                 

                 print(f"Error: Could not even find index for '{target_column_name}'. One-hot vector will be all zeros for site.")
    
    else:
        site_to_encode = 'unknown'
        target_column_name = f'site_{site_to_encode}'
        # Try setting 'unknown' index
        try:
            site_index = ANATOM_SITE_COLUMNS_TRAIN.index(target_column_name)
            site_onehot_vector[site_index] = 1.0
        except ValueError:
             print(f"Error: Could not find index for '{target_column_name}' (unknown). One-hot vector will be all zeros for site.")


    feature_vector = np.concatenate(
        ([sex_feature], [age_processed], site_onehot_vector)
    ).astype(np.float32) # Final check for float32 type

    return feature_vector


class TumorDataset (Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        # Kept original listing style, added .dcm filter implicitly needed
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.dcm')])
        self.transform = transform
        self.length = len(self.file_names)


    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.file_names[idx])
        ds = pydicom.dcmread(path)
        name = self.file_names[idx][:-4]

        array = ds.pixel_array
        array = array.astype(np.uint8)
        if self.transform:
            image = self.transform(image=array)["image"] 

        
        metadata = preprocess_single_inference_data(ds.PatientSex,ds.PatientAge,ds.BodyPartExamined)

        return image, metadata, name

    def __len__(self):
        return self.length
    

image_size = 224 
test_transforms = A.Compose([
    A.Resize(image_size,image_size),
    A.Normalize(),
    ToTensorV2()
])



def image_model_inference(model,test_loader,thresh= 0.3,if_metadata = 0):
    all_logits = []
    all_probs = []
    all_labels = []
    all_names = []
    with torch.no_grad():
        for batch_index, (image,features,name) in enumerate(test_loader):

            #print(features.shape)
            image =image.to(device)
            features =features.to(device)
            if(if_metadata):
                outputs = model(image,features)        
            else:
                outputs = model(image)

            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs.view(-1))  # binary case
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1] 

            labels = list((probs.cpu().numpy().flatten() > thresh).astype(int))
            probs = list(probs.cpu().numpy().flatten())
            all_probs+=probs
            all_labels += labels
            all_names+=(list(name))
            if len(all_names) >= len(test_loader.dataset):
                break


    
    df = pd.DataFrame({"image_name":all_names, "target": all_labels})
    return all_names,all_probs,all_labels, df


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


model_paths= ["modeli/model_1"]
which_model = [0]
if_metadata = [1]
ANATOM_SITE_COLUMNS_TRAIN = list(np.load("anatom_site_columns.npy"))
MEDIAN_AGE= 50
THRESHOLD = 0.3

## Tu potrebno odabrati path
folder_path = "train"
# folder_path = "sys.arg[1]"
output_path = "csv_rj.csv"
# folder_path = "sys.arg[2]"

test_dataset = TumorDataset(folder_path, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_probs = np.zeros((len(model_paths),800))#test_dataset.length
for i, weights_file_path_relative in enumerate(model_paths):

    absolute_path = os.path.abspath(weights_file_path_relative)
    state_dict = torch.load(absolute_path, map_location=device)
    if(which_model[i] == 1):
        model = EfficientNetB6BinaryClassifier()#tu dodati metadata
    if(which_model[i] == 0):
        if(if_metadata[i]):
            model = ConvNeXtBinaryClassifier_metadata()
        else:
            model = ConvNeXtBinaryClassifier()
    
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    names,probs,labels, df = image_model_inference(model, test_loader, THRESHOLD, if_metadata[i])
    
    all_probs[i,:] = np.array(probs)

np.mean(all_probs, axis = 0)

labels = list((all_probs.flatten() > THRESHOLD).astype(int))
    
df = pd.DataFrame({"image_name":names, "target": labels})
df.to_csv(output_path)

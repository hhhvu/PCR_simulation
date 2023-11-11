import lightning.pytorch as pl
import torchvision, torch, torchio as tio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import numpy as np
import json
import tqdm
import os
import pickle as pkl
import pandas as pd
from PIL import Image

class ImageSequenceDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        mean_list = []
        std_list = []

        for key, curve in tqdm(self.curve_dict_train.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)

        norm_mean = np.array(mean_list).mean().item()
        norm_std = np.array(std_list).mean().item()

        self.train = ImageSequenceDataset(self.curve_dict_train, self.target_df_train, mean=norm_mean, std=norm_std)
        self.val = ImageSequenceDataset(self.curve_dict_val, self.target_df_val, mean=norm_mean, std=norm_std)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

class ImageSequenceDataset(Dataset):
    def __init__(self, curve_dict, target_df, img_directory = 'data/curve_imgs/', sequence_len=40, 
                 mean=0, std=1):
        self.curve_dict = curve_dict
        self.target_df = target_df

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.img_directory = img_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((128, 128)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])
   
    def __len__(self):
        return len(self.curve_dict.keys())
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]

        # Image processing
        curve_img_path = os.path.join(self.img_directory, f'curve_{curve_idx}.png')
        curve_img = Image.open(curve_img_path)
        curve_img = self.img_transforms(curve_img)

        #sequence processing
        sequence = self.curve_dict[curve_idx][:self.sequence_len]
        #TODO fix normalization to normalizing by mean and std of sequences in train set
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence_normalized = (sequence - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)

        return curve_img, sequence_normalized
    

class ImageSequenceGeneDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        mean_list = []
        std_list = []

        for key, curve in tqdm(self.curve_dict_train.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)

        norm_mean = np.array(mean_list).mean().item()
        norm_std = np.array(std_list).mean().item()

        self.train = ImageSequenceGeneDataset(self.curve_dict_train, self.target_df_train, mean=norm_mean, std=norm_std)
        self.val = ImageSequenceGeneDataset(self.curve_dict_val, self.target_df_val, mean=norm_mean, std=norm_std)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

class ImageSequenceGeneDataset(Dataset):
    def __init__(self, curve_dict, target_df, img_directory = 'data/curve_imgs/', sequence_len=40, 
                 mean=0, std=1):
        self.curve_dict = curve_dict
        self.target_df = target_df

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.img_directory = img_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((128, 128)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])
   
    def __len__(self):
        return len(self.curve_dict.keys())
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]

        # Image processing
        curve_img_path = os.path.join(self.img_directory, f'curve_{curve_idx}.png')
        curve_img = Image.open(curve_img_path)
        curve_img = self.img_transforms(curve_img)

        #sequence processing
        sequence = self.curve_dict[curve_idx][:self.sequence_len]
        #TODO fix normalization to normalizing by mean and std of sequences in train set
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence_normalized = (sequence - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)

        #gene info processing
        row = self.target_df.loc[self.target_df['curve_idx'] == curve_idx]
        gene_type = torch.tensor(row[self.one_hot.columns].values, dtype=torch.float32)

        return curve_img, sequence_normalized, gene_type
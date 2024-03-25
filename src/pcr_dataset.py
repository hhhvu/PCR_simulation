import lightning.pytorch as pl
import torchvision, torch, torchio as tio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
import numpy as np
import json
from tqdm import tqdm
import os
import pickle as pkl
import pandas as pd
from PIL import Image
from torch.utils.data import WeightedRandomSampler

class ImageSequenceDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, img_directory, batch_size=32, shuffle=True, num_workers =16, igi_call=False):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.igi_call = igi_call
        self.img_directory = img_directory

        print("WE ARE USING THE IMAGE SEQUENCE DATASET")

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df['igi_fp'] = (self.target_df['Igi_call_quant'] > self.target_df['groundtruth_target']).astype(int)
        self.target_df['igi_fn'] = (self.target_df['Igi_call_quant'] < self.target_df['groundtruth_target']).astype(int)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        self.target_df_test = self.target_df[self.target_df['split']=='test']
        self.curve_dict_test = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_test['curve_idx'].values}

        mean_list = []
        std_list = []
        deltas_list = []

        for key, curve in tqdm(self.curve_dict_train.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            delta_curve = np.array(curve).max().item() - np.array(curve).min().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)
            deltas_list.append(delta_curve)

        self.norm_mean = np.array(mean_list).mean().item()
        self.norm_std = np.array(std_list).mean().item()

        #print(delta_curve)
        # print(np.array(deltas_list).shape)
        # print(np.array(deltas_list).std())

        self.delta_mean = np.array(deltas_list).mean().item()
        self.delta_std = np.array(deltas_list).std().item()
        # print('delta std and mean')

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = ImageSequenceDataset(self.curve_dict_train, self.target_df_train, self.img_directory, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std,
                                          delta_mean=self.delta_mean, delta_std=self.delta_std)
        self.val = ImageSequenceDataset(self.curve_dict_val, self.target_df_val, self.img_directory, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std,
                                        delta_mean=self.delta_mean, delta_std=self.delta_std)
        self.test = ImageSequenceDataset(self.curve_dict_test, self.target_df_test, self.img_directory, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std,
                                         delta_mean=self.delta_mean, delta_std=self.delta_std)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

class ImageSequenceDataset(Dataset):
    def __init__(self, curve_dict, target_df, img_directory = 'data/curve_imgs/', sequence_len=40, igi_call=True,
                 mean=0, std=1, delta_mean=0, delta_std=1):
        self.curve_dict = curve_dict
        self.target_df = target_df

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.img_directory = img_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len
        self.igi_call = igi_call

        self.mean = mean
        self.std = std

        self.delta_mean = delta_mean
        self.delta_std = delta_std 

        # Image transformations: Resize and Normalize

        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((224, 224)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])

        # self.img_transforms = transforms.Compose([
        #     transforms.Lambda(lambda image: image.convert('RGB')),
        #     transforms.Resize((128, 128)),  # Resizing to a consistent size
        #     transforms.ToTensor(),  # Convert PIL image to tensor
        #     transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
        #     ])
   
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
        delta = torch.max(sequence) - torch.min(sequence)
        # print(f"Delta: {delta}")
        # print(f"delta_mean: {self.delta_mean}")
        # print(f"delta_std: {self.delta_std}")
        delta_norm = (delta - torch.tensor(self.delta_mean, dtype=torch.float32)) / torch.tensor(self.delta_std, dtype=torch.float32)
        # print(f"delta_norm: {delta_norm}")

        row = self.target_df.loc[self.target_df['curve_idx'] == curve_idx]
        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.float)

        if self.igi_call:
            igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.float)
            igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.float)
            target = torch.stack([target, igi_fp, igi_fn], dim=0)

        return (curve_img, delta_norm), target
    
class ImageDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, batch_size=32, shuffle=True, resampling=False, num_workers=4, igi_call=False, gen_preds=False, img_directory = 'data/curve_imgs_new/', external=False):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resampling = resampling
        self.num_workers = num_workers
        self.igi_call = igi_call
        self.gen_preds = gen_preds
        self.img_directory = img_directory
        self.external = external

        print("WE ARE USING THE IMAGE DATASET")

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df['igi_fp'] = (self.target_df['Igi_call_quant'] > self.target_df['groundtruth_target']).astype(int)
        self.target_df['igi_fn'] = (self.target_df['Igi_call_quant'] < self.target_df['groundtruth_target']).astype(int)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        self.target_df_test = self.target_df[self.target_df['split']=='test']
        self.curve_dict_test = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_test['curve_idx'].values}

        mean_list = []
        std_list = []

        if self.external:
            rotation_dict = self.curve_dict_test
        else:
            rotation_dict = self.curve_dict_train

        for key, curve in tqdm(rotation_dict.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)

        self.norm_mean = np.array(mean_list).mean().item()
        self.norm_std = np.array(std_list).mean().item()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = ImageDataset(self.curve_dict_train, self.target_df_train, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds, img_directory = self.img_directory)
        self.val = ImageDataset(self.curve_dict_val, self.target_df_val, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds, img_directory = self.img_directory)
        self.test = ImageDataset(self.curve_dict_test, self.target_df_test, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds, img_directory = self.img_directory)

    def train_dataloader(self):
        sampler = None
        
        if self.resampling:
            shuffle = False
            y = pd.Series(self.train.target_df['groundtruth_target'])
            class_counts = y.value_counts()  # class counts

            n_samples = len(y)

            sample_weights = n_samples / np.array([class_counts[cl_idx]
                                                for cl_idx in y])
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
                                            replacement=True)
            
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

class ImageDataset(Dataset):
    def __init__(self, curve_dict, target_df, img_directory = 'data/curve_imgs_new/', sequence_len=40, igi_call=True,
                 mean=0, std=1, gen_preds=False):
        
        self.curve_dict = curve_dict
        self.target_df = target_df
        self.gen_preds = gen_preds

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.img_directory = img_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std
        self.igi_call = igi_call

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((224, 224)),  # Resizing to a consistent size
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

        #gene info processing
        row = self.target_df.loc[self.target_df['curve_idx'] == curve_idx]

        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.float)

        if self.igi_call:
            igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.float)
            igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.float)
            target = torch.stack([target, igi_fp, igi_fn], dim=0)

        if self.gen_preds:
            return [curve_img], target, curve_idx
        else:
            return [curve_img], target #, curve_idx

class ImageSequenceGeneDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, img_directory, batch_size=32, shuffle=True, resampling=False, num_workers=4, igi_call=False):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resampling = resampling
        self.num_workers = num_workers
        self.igi_call = igi_call
        
        self.img_directory = img_directory

        print("WE ARE USING THE IMAGE SEQUENCE GENE DATASET")

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df['igi_fp'] = (self.target_df['Igi_call_quant'] > self.target_df['groundtruth_target']).astype(int)
        self.target_df['igi_fn'] = (self.target_df['Igi_call_quant'] < self.target_df['groundtruth_target']).astype(int)

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

        self.norm_mean = np.array(mean_list).mean().item()
        self.norm_std = np.array(std_list).mean().item()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = ImageSequenceGeneDataset(self.curve_dict_train, self.target_df_train, self.img_directory,igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std)
        self.val = ImageSequenceGeneDataset(self.curve_dict_val, self.target_df_val, self.img_directory, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

class ImageSequenceGeneDataset(Dataset):
    def __init__(self, curve_dict, target_df, img_directory = 'data/curve_imgs/', sequence_len=40, igi_call=False,
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
        self.igi_call = igi_call

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((224, 224)),  # Resizing to a consistent size
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

        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.float)

        if self.igi_call:
            igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.float)
            igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.float)
            target = torch.stack([target, igi_fp, igi_fn], dim=0)

        return (curve_img, sequence_normalized.unsqueeze(1), gene_type.squeeze(1)), target

class SequenceDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Image+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, img_directory=None, batch_size=32, shuffle=True, resampling=False, num_workers=4, igi_call=False, gen_preds=False, external=False):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resampling = resampling
        self.num_workers = num_workers
        self.igi_call = igi_call
        self.gen_preds = gen_preds
        self.external = external

        print("WE ARE USING THE SEQUENCE DATASET")

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df['igi_fp'] = (self.target_df['Igi_call_quant'] > self.target_df['groundtruth_target']).astype(int)
        self.target_df['igi_fn'] = (self.target_df['Igi_call_quant'] < self.target_df['groundtruth_target']).astype(int)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        self.target_df_test = self.target_df[self.target_df['split']=='test']
        self.curve_dict_test = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_test['curve_idx'].values}

        mean_list = []
        std_list = []

        if self.external:
            rotation_dict = self.curve_dict_test
        else:
            rotation_dict = self.curve_dict_train

        for key, curve in tqdm(rotation_dict.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)

        self.norm_mean = np.array(mean_list).mean().item()
        self.norm_std = np.array(std_list).mean().item()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        print("SETUP DONE")
        self.train = SequenceDataset(self.curve_dict_train, self.target_df_train, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)
        self.val = SequenceDataset(self.curve_dict_val, self.target_df_val, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)
        self.test = SequenceDataset(self.curve_dict_test, self.target_df_test, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)

    def train_dataloader(self):
        sampler = None
        
        if self.resampling:
            shuffle = False
            y = pd.Series(self.train.target_df['groundtruth_target'])
            class_counts = y.value_counts()  # class counts

            n_samples = len(y)

            sample_weights = n_samples / np.array([class_counts[cl_idx]
                                                for cl_idx in y])
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
                                            replacement=True)
            
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

class SequenceDataset(Dataset):
    def __init__(self, curve_dict, target_df, sequence_len=40, igi_call=False, mean=0, std=1, gen_preds=False): #img_directory = 'data/curve_imgs/',
        self.curve_dict = curve_dict
        self.target_df = target_df

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        #self.img_directory = img_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std
        self.igi_call = igi_call
        self.gen_preds = gen_preds

        if np.isnan(self.mean):
            # self.mean, self.std = 155626.8370536778, 94477.0057018847
            mean_list = []
            std_list = []

            for key, curve in tqdm(self.curve_dict.items()):
                mean_curve = np.array(curve).mean().item()
                std_curve = np.array(curve).std().item()

                mean_list.append(mean_curve)
                std_list.append(std_curve)

            self.mean = np.array(mean_list).mean().item()
            self.std = np.array(std_list).mean().item()

            print(self.mean, self.std)

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((224, 224)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])
   
    def __len__(self):
        return len(self.curve_dict.keys())
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]

        # Image processing
        # curve_img_path = os.path.join(self.img_directory, f'curve_{curve_idx}.png')
        # curve_img = Image.open(curve_img_path)
        # curve_img = self.img_transforms(curve_img)

        #sequence processing
        sequence = self.curve_dict[curve_idx][:self.sequence_len]
        #TODO fix normalization to normalizing by mean and std of sequences in train set
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence_normalized = (sequence - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)

        #gene info processing
        row = self.target_df.loc[self.target_df['curve_idx'] == curve_idx]
        # gene_type = torch.tensor(row[self.one_hot.columns].values, dtype=torch.float32)
        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.float)

        if self.igi_call:
            igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.float)
            igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.float)
            target = torch.stack([target, igi_fp, igi_fn], dim=0)

        if self.gen_preds:
            return [sequence_normalized.unsqueeze(1)], target, curve_idx
        else:
            return [sequence_normalized.unsqueeze(1)], target


class SequenceGeneDataModule(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for Gene+Sequence dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, curve_dict_path, target_df_path, img_directory=None, batch_size=32, shuffle=True, resampling=False, num_workers=4, igi_call=False, gen_preds=False, external=False):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resampling = resampling
        self.num_workers = num_workers
        self.igi_call = igi_call
        self.gen_preds = gen_preds
        self.external = external

        print("WE ARE USING THE SEQUENCE GENE DATASET")

        with open(curve_dict_path, 'rb') as file:
            self.curve_dict = pkl.load(file)
        self.target_df = pd.read_csv(target_df_path)

        self.target_df['igi_fp'] = (self.target_df['Igi_call_quant'] > self.target_df['groundtruth_target']).astype(int)
        self.target_df['igi_fn'] = (self.target_df['Igi_call_quant'] < self.target_df['groundtruth_target']).astype(int)

        self.target_df_train = self.target_df[self.target_df['split']=='train']
        self.curve_dict_train = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_train['curve_idx'].values}
        
        self.target_df_val = self.target_df[self.target_df['split']=='val']
        self.curve_dict_val = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_val['curve_idx'].values}

        self.target_df_test = self.target_df[self.target_df['split']=='test']
        self.curve_dict_test = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df_test['curve_idx'].values}


        mean_list = []
        std_list = []

        if self.external:
            rotation_dict = self.curve_dict_test
        else:
            rotation_dict = self.curve_dict_train

        for key, curve in tqdm(rotation_dict.items()):
            mean_curve = np.array(curve).mean().item()
            std_curve = np.array(curve).std().item()

            mean_list.append(mean_curve)
            std_list.append(std_curve)

        self.norm_mean = np.array(mean_list).mean().item()
        self.norm_std = np.array(std_list).mean().item()

    def prepare_data(self):
        return

    def setup(self, stage=None):
        self.train = SequenceGeneDataset(self.curve_dict_train, self.target_df_train, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)
        self.val = SequenceGeneDataset(self.curve_dict_val, self.target_df_val, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)
        self.test = SequenceGeneDataset(self.curve_dict_test, self.target_df_test, igi_call=self.igi_call, mean=self.norm_mean, std=self.norm_std, gen_preds=self.gen_preds)

    def train_dataloader(self):
        sampler = None
        
        if self.resampling:
            shuffle = False
            y = pd.Series(self.train.target_df['groundtruth_target'])
            class_counts = y.value_counts()  # class counts

            n_samples = len(y)

            sample_weights = n_samples / np.array([class_counts[cl_idx]
                                                for cl_idx in y])
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(sample_weights),
                                            replacement=True)
            
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)

class SequenceGeneDataset(Dataset):
    def __init__(self, curve_dict, target_df, sequence_len=40, igi_call=False,
                 mean=0, std=1, gen_preds=False):
        self.curve_dict = curve_dict
        self.target_df = target_df
        self.gen_preds = gen_preds

        # List of desired columns in specific order
        columns_order = ['target_E gene', 'target_MS2', 'target_N gene', 
                        'target_ORF1ab', 'target_RnaseP', 'target_S gene']

        # Perform one-hot encoding
        one_hot_encoded = pd.get_dummies(self.target_df['target'], prefix='target', dtype=int)

        # Ensure all desired columns are present, even if some categories might be missing in the data
        # This step fills in missing columns with 0s
        for column in columns_order:
            if column not in one_hot_encoded.columns:
                one_hot_encoded[column] = pd.Series(0, index=one_hot_encoded.index, dtype=int)

        # Reorder the columns to match the specified order
        self.one_hot = one_hot_encoded[columns_order]

        print(self.one_hot.head(10))
        
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std
        self.igi_call = igi_call

        if np.isnan(self.mean):
            # self.mean, self.std = 155626.8370536778, 94477.0057018847
            mean_list = []
            std_list = []

            for key, curve in tqdm(self.curve_dict.items()):
                mean_curve = np.array(curve).mean().item()
                std_curve = np.array(curve).std().item()

                mean_list.append(mean_curve)
                std_list.append(std_curve)

            self.mean = np.array(mean_list).mean().item()
            self.std = np.array(std_list).mean().item()

            print(self.mean, self.std)
   
    def __len__(self):
        return len(self.curve_dict.keys())
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]

        #sequence processing
        sequence = self.curve_dict[curve_idx][:self.sequence_len]
        #TODO fix normalization to normalizing by mean and std of sequences in train set
        sequence = torch.tensor(sequence, dtype=torch.float32)
        sequence_normalized = (sequence - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(self.std, dtype=torch.float32)

        #gene info processing
        row = self.target_df.loc[self.target_df['curve_idx'] == curve_idx]
        gene_type = torch.tensor(row[self.one_hot.columns].values, dtype=torch.float32)

        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.float)

        if self.igi_call:
            igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.float)
            igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.float)
            target = torch.stack([target, igi_fp, igi_fn], dim=0)

        if self.gen_preds:
            return (sequence_normalized.unsqueeze(1), gene_type.squeeze(1)), target, curve_idx
        else:
            return (sequence_normalized.unsqueeze(1), gene_type.squeeze(1)), target
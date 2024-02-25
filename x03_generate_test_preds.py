# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import io, os
from tqdm import tqdm

from PIL import Image
from torchvision import models 
from torchvision.models import resnet18

class ImageSequenceGeneDataset(Dataset):
    def __init__(self, curve_dict, target_df,img_directory = 'data/curve_imgs/', split='train', train=True, sequence_len=40, 
                 mean=0, std=1):
        self.curve_dict = curve_dict
        self.target_df = target_df

        #one-hot encode gene indicator
        self.one_hot = pd.get_dummies(self.target_df['target'], prefix='target')
        self.target_df = pd.concat([self.target_df, self.one_hot], axis=1)

        self.img_directory = img_directory
        self.train = train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_len = sequence_len

        self.mean = mean
        self.std = std

        # Image transformations: Resize and Normalize
        self.img_transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.Resize((224, 224)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])

        #Implementation of train test split
        self.target_df = self.target_df[self.target_df['split']==split]
        self.curve_dict = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df['curve_idx'].values}
        
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

        #target data retrieval
        # Extract values from the dataframe
        target = torch.tensor(row['groundtruth_target'].values[0], dtype=torch.long)
        igi_fp = torch.tensor(row['igi_fp'].values[0], dtype=torch.long)
        igi_fn = torch.tensor(row['igi_fn'].values[0], dtype=torch.long)

        # Create a 3-dimensional vector
        vector = [target, igi_fp, igi_fn]
        #target = self.target_df.loc[self.target_df['curve_idx'] == curve_idx, 'groundtruth_target'].values[0]

        return curve_img, sequence_normalized, gene_type, vector, curve_idx

class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, sequence_length, num_layers=5, genes = 6, num_heads=3, delta=16):
        super(FusionModel, self).__init__()

        self.latent_dim = latent_dim
        self.delta = delta
        
        self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  # Adjusting to output a 512-dimensional 

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Delta Sequence processing via LSTM
        # self.lstm_delta = nn.LSTM(input_size-1, hidden_size, num_layers, batch_first=True)
        # self.hidden_state_delta = (torch.zeros(num_layers, sequence_length-1, hidden_size), torch.zeros(num_layers, sequence_length-1, hidden_size))
        # # Final fully connected layer to ensure the LSTM output has a size of 512
        # self.lstm_fc_delta = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes
        neural_net_input = self.latent_dim*2 + genes + delta #(self.latent_dim-1)
        print(neural_net_input)

        # Fusion of image and sequence representations
        self.fc = nn.Sequential(
            nn.Linear(neural_net_input, 512),  # Concatenated vectors are of size 1024 (512 from image + 512 from sequence)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
            # nn.Linear(64, 1),
            # nn.Sigmoid()
        )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])


    def forward(self, image, sequence, genes):
        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Calculating delta
        delta_latent = torch.max(sequence, dim=1)[0] - torch.min(sequence, dim=1)[0]
        delta_latent = delta_latent.expand((-1, self.delta))
        # delta_seq = sequence[:, 1:] - sequence[:, :-1] #taking first difference
        # lstm_out_delta, _ = self.lstm_delta(delta_seq)
        # seq_latent_delta = self.lstm_fc_delta(lstm_out_delta[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), delta_latent), dim=1) #seq_latent_delta
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = [torch.sigmoid(head(output)) for head in self.heads]

        return outputs

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.version.cuda)
print(torch.__version__)
print(device)

if __name__ == "__main__":

    with open('data/groundtruth_df_curve_dict_split_v2.pkl', 'rb') as file:
        curve_dict = pkl.load(file)
    
    target_df = pd.read_csv('data/groundtruth_df_target_data_split_v2.csv')  # Load your DataFrame here
    target_df.loc[:,['groundtruth_target']] = 1*(target_df.groundtruth == 1)

    #Define two error targets
    target_df.loc[:,'Igi_call_quant'] = 1*(target_df.igi_call == 'Positive')
    target_df['igi_fp'] = (target_df['Igi_call_quant'] > target_df['groundtruth_target']).astype(int)
    target_df['igi_fn'] = (target_df['Igi_call_quant'] < target_df['groundtruth_target']).astype(int)

    print(target_df.head(10))
    print(target_df.columns)
    print(target_df.split.unique())

    print(target_df[target_df['split']=='test'].shape)

    ###########################################
    ## Get the right normalization values
    ###########################################
    
    target_df_filtered = target_df[target_df['split']=='train']
    curve_dict_filtered = {k: curve_dict[k] for k in curve_dict.keys() if k in target_df_filtered['curve_idx'].values}

    mean_list = []
    std_list = []

    for key, curve in tqdm(curve_dict_filtered.items()):
        mean_curve = np.array(curve).mean().item()
        std_curve = np.array(curve).std().item()

        mean_list.append(mean_curve)
        std_list.append(std_curve)

    norm_mean = np.array(mean_list).mean().item()
    norm_std = np.array(std_list).mean().item()

    ###########################################
    ####### Create dataset 

    testvit_dataset = ImageSequenceGeneDataset(curve_dict, target_df,img_directory = 'data/curve_imgs_v2/', split='val', sequence_len=40,
                                    mean=norm_mean, std = norm_std)
    testvit_loader = DataLoader(testvit_dataset, batch_size=32, pin_memory=True, shuffle=True)


    ###########################################
    ## Initialize model
    ###########################################

    # Adjust the parameters as per your needs
    sequence_length = 40  # Suppose the length of your sequence is 100
    input_size = 1  # Number of input features per sequence element
    hidden_size = 512
    latent_dim = 512
    num_layers = 3
    num_epoch = 50
    genes = len(target_df['target'].unique())
    #delta_size = 512
    delta_size = 64

    model = FusionModel(input_size, hidden_size, latent_dim, sequence_length, num_layers=num_layers, genes=genes, delta=delta_size)
    model.load_state_dict(torch.load('output/10_27_fusion_model_vit_delta64.pth'))
    model.to(device)  # If you are using GPU

    ##########################################
    ### Calculate outputs
    ##########################################

    probs, igi_fp, igi_fn = [], [], []
    curve_ids = []
    model.eval()
    for batch in tqdm(testvit_loader):
        images, sequences, gene, labels, ids = batch

        images = images.to(device)
        sequences = sequences.to(device).unsqueeze(2)
        gene = gene.to(device)

        out = model(images, sequences, gene)
        probs.append(out[0].detach().cpu().numpy())
        igi_fp.append(out[1].detach().cpu().numpy())
        igi_fn.append(out[2].detach().cpu().numpy())

        curve_ids.append(ids)
    
    test_pred_df = pd.DataFrame({'curve_idx': np.concatenate(curve_ids), 
                             'outputs': np.concatenate(probs).squeeze(),
                             'igi_fp': np.concatenate(igi_fp).squeeze(),
                             'igi_fn': np.concatenate(igi_fn).squeeze()})
    test_pred_df.to_csv('data/model_outputs/10_27_fusion_model_vit_delta64_val_pred_df.csv', index = False)
    

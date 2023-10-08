from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import pickle as pkl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import io, os
from tqdm import tqdm

from torchvision.models import resnet18

class PlotImageDataset(Dataset):
    def __init__(self, curve_dict, target_df, save_plots = False):
        self.curve_dict = curve_dict
        self.target_df = target_df
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_plots = save_plots
        
    def __len__(self):
        return len(self.curve_dict)
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]
        sequence = self.curve_dict[curve_idx][:40]
        target = self.target_df.loc[self.target_df['curve_idx'] == curve_idx, 'groundtruth_target'].values[0]

        if self.save_plots:
            if not os.path.exists('data/curve_imgs/'):
                os.makedirs('data/curve_imgs/')
            plt.plot(sequence, linewidth=6)
            plt.savefig(f'data/curve_imgs/curve_{curve_idx}.png')

        # TODO: LOAD IMAGES FROM IMAGE FOLDER AND CHANGE OUTPUT OF GETITEM
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


class SequencesDataset(Dataset):
    def __init__(self, curve_dict, target_df, save_plots = False):
        self.curve_dict = curve_dict
        self.target_df = target_df
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def __len__(self):
        return len(self.curve_dict)
    
    def __getitem__(self, idx):
        curve_idx = list(self.curve_dict.keys())[idx]
        sequence = self.curve_dict[curve_idx][:40]
        target = self.target_df.loc[self.target_df['curve_idx'] == curve_idx, 'groundtruth_target'].values[0]
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class ConvLSTM(nn.Module):
    def __init__(self, input_size, conv_out_channels, kernel_size, hidden_size, output_size, seq_len):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=conv_out_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=conv_out_channels, out_channels=conv_out_channels, kernel_size=kernel_size, padding=1)
        self.lstm = nn.LSTM(input_size=conv_out_channels*(seq_len-kernel_size+1), hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # swap sequence length and feature dimension for Conv1D
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # swap back after Conv1D
        x = x.reshape(x.size(0), 1, -1)  # merge the sequence length and feature dimension for LSTM
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    ###########################################
    ## Load the data
    ###########################################

    plt.rcParams["figure.autolayout"] = True
    plt.grid(False)
    plt.axis('off')

    with open('data/groundtruth_df_curve_dict.pkl', 'rb') as file:
        curve_dict = pkl.load(file)
    target_df = pd.read_csv('data/groundtruth_df_target_data.csv')  # Load your DataFrame here

    # Create Dataset and DataLoader
    # dataset = SequencesDataset(curve_dict, target_df)
    dataset = PlotImageDataset(curve_dict, target_df, save_plots=False) # Use save_plots to create pngs

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

    '''
    train_curve_idxs = []
    val_curve_idxs = []
    for idx, _, _ in train_dataset:
        train_curve_idxs.append(idx)
    for idx, _, _ in val_dataset:
        val_curve_idxs.append(idx)
    
    print(len(dataset))
    print(len(train_curve_idxs))
    print(len(val_curve_idxs))
    curve_idxs_dict = {'train': train_curve_idxs, 'val': val_curve_idxs}

    with open('dataset_curve_idxs.pkl', 'wb') as f:
        pkl.dump(curve_idxs_dict, f)
    '''

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


    ###########################################
    ## Initialize model
    ###########################################

    #Simple model
    # model = SimpleLSTM(input_size=10, hidden_size=50, output_size=2)  # Adjust the parameters as per your needs
    # model.to('cuda')  # If you are using GPU

    # Adjust the parameters as per your needs
    seq_len = 40  # Suppose the length of your sequence is 100
    input_size = 1  # Number of input features per sequence element
    model = ConvLSTM(input_size=input_size, conv_out_channels=32, kernel_size=3, hidden_size=50, output_size=2, seq_len=seq_len)
    model.to(device)  # If you are using GPU

    # model = nn.Sequential(resnet18(), nn.Flatten(), nn.Linear(200, 2))
    # model.to(device)

    ###########################################
    ## Set up metric logging
    ###########################################

    experiment = Experiment(
        api_key="7smwpzl0FeZJcESqBITDniX7I",
        project_name="classification-arch-search",
        workspace="pcr-simulation"
    )

    experiment.set_name('ConvLSTM_MetricTest')

    ###########################################
    ## Training Loop
    ###########################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float('inf')

    print('Device:', device)

    for name, layer in model.named_modules():
        print(name)
        # layer.register_backward_hook(lambda module, grad_input, grad_output: print(grad_output))

    for epoch in range(50):  # Choose the number of epochs
        model.train()
        running_loss = 0.0

        print('Starting epoch', epoch)
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)

            # print(sequences.shape)
            # print(labels.shape)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        corrects = 0

        val_pred_labels, val_true_labels = [], []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sequences.size(0)
                corrects += (outputs.argmax(1) == labels).sum().item()

                val_pred_labels.extend(outputs.argmax(1) == labels)
                val_true_labels.extend(labels)
                
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            log_model(experiment, model, model_name=f"CurModel_{epoch}")

        print(len(val_pred_labels))
        print(len(val_true_labels))

        val_acc, avg_train_loss, avg_val_loss = corrects/len(val_loader.dataset), running_loss/len(train_loader.dataset), val_loss/len(val_loader.dataset)
        val_auc = roc_auc_score(val_true_labels, val_pred_labels)

        experiment.log_metrics({'Validation Accuracy': val_acc, 'Avg Training Loss': avg_train_loss, 'Avg Validation Loss': avg_val_loss, 'Validation AUC': val_auc}, epoch=epoch)
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_acc}, Validation AUC: {val_auc}")


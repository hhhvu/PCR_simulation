from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

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
    def __init__(self, curve_dict, target_df,img_directory = 'data/curve_imgs/', train=True, sequence_len=40, 
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
            transforms.Resize((128, 128)),  # Resizing to a consistent size
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalizing to [0,1]
            ])

        #Implementation of train test split
        if self.train:
            self.target_df = self.target_df[self.target_df['split']=='train']
            self.curve_dict = {k: self.curve_dict[k] for k in self.curve_dict.keys() if k in self.target_df['curve_idx'].values}
        else:
            self.target_df = self.target_df[self.target_df['split']=='val']
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

        return curve_img, sequence_normalized, gene_type, vector

class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, sequence_length, num_layers=5, genes = 6, num_heads=3):
        super(FusionModel, self).__init__()

        self.latent_dim = latent_dim
        
        # Image processing via EfficientNet_V2_L
        # TODO change to true
        self.effnet = models.efficientnet_v2_l(pretrained=True)
        num_ftrs = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Linear(num_ftrs, self.latent_dim)  # Adjusting to output a 512-dimensional 

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes
        neural_net_input = self.latent_dim*2 + genes

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
        img_latent = self.effnet(image)

        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1)), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = [torch.sigmoid(head(output)) for head in self.heads]

        return outputs

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.version.cuda)
print(torch.__version__)
print(device)

# Command to send job to Wynton GPUs
#qsub -cwd -q gpu.q -l h_rt=08:30:00,gpu_mem=12000M FusionModel_script.sh

#Make sure to install CUDA version compatible with GPU drivers installed on wynton
#conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch

#To export models run in local terminal
#scp alex_schubert@plog1.wynton.ucsf.edu:/wynton/protected/home/ibrahim/alex_schubert/PCR_simulation/output/fusion_model/best_model_no_pretrain.pth ~/Alex/Studium/01_UC_Berkeley/01_Research_Projects/PCR_simulation/PCR_simulation/

if __name__ == "__main__":

    MODEL_SAVE_PATH = 'output/10_21_fusion_model_gene'

    if not os.path.isdir(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    ###########################################
    ## Load the data
    ###########################################

    plt.rcParams["figure.autolayout"] = True
    plt.grid(False)
    plt.axis('off')

    with open('data/groundtruth_df_curve_dict_split_v2.pkl', 'rb') as file:
        curve_dict = pkl.load(file)
    target_df = pd.read_csv('data/groundtruth_df_target_data_split_v2.csv')  # Load your DataFrame here

    #Define two error targets
    target_df['igi_fp'] = (target_df['Igi_call_quant'] > target_df['groundtruth_target']).astype(int)
    target_df['igi_fn'] = (target_df['Igi_call_quant'] < target_df['groundtruth_target']).astype(int)

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
    ## Save curves as images
    ###########################################

    # for idx in  tqdm(range(len(curve_dict.keys()))):
    #     curve_idx = list(curve_dict.keys())[idx]
    #     sequence = curve_dict[curve_idx][:40]

    #     if not os.path.exists('data/curve_imgs_axis/'):
    #         os.makedirs('data/curve_imgs_axis/')
    #     plt.plot(sequence, linewidth=6)
    #     #plt.axis('off')  # This will turn off the axis labels and ticks
    #     plt.axis('on') 
    #     plt.show()
    #     plt.savefig(f'data/curve_imgs_axis/curve_{curve_idx}.png')
    #     plt.clf()

    ###########################################
    ## Set-up data objects
    ###########################################

    # TODO extract mean and std of sequences in train dataset

    # Create Dataset and DataLoader
    train_dataset = ImageSequenceGeneDataset(curve_dict, target_df,img_directory = 'data/curve_imgs/', train=True, sequence_len=40,
                                         mean=norm_mean, std = norm_std)
    val_dataset = ImageSequenceGeneDataset(curve_dict, target_df,img_directory = 'data/curve_imgs/', train=False, sequence_len=40,
                                       mean=norm_mean, std = norm_std)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


    ###########################################
    ## Initialize model
    ###########################################

    #Simple model
    # model = SimpleLSTM(input_size=10, hidden_size=50, output_size=2)  # Adjust the parameters as per your needs
    # model.to('cuda')  # If you are using GPU

    # Adjust the parameters as per your needs
    sequence_length = 40  # Suppose the length of your sequence is 100
    input_size = 1  # Number of input features per sequence element
    hidden_size = 512
    latent_dim = 512
    num_layers = 3
    num_epoch = 10
    genes = len(target_df['target'].unique())

    model = FusionModel(input_size, hidden_size, latent_dim, sequence_length, num_layers=num_layers, genes=genes)
    # model = ConvLSTM(input_size=input_size, conv_out_channels=32, kernel_size=3, hidden_size=50, output_size=2, seq_len=seq_len)
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

    experiment.set_name('FusionModel_ImgSeqGene')

    ###########################################
    ## Training Loop
    ###########################################

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float('inf')

    print('Device:', device)

    val_losses, train_losses, val_aucs = [], [], []

    for epoch in tqdm(range(num_epoch)):  # Choose the number of epochs
        model.train()
        running_loss = 0.0

        print('Starting epoch', epoch)
        
        for images, sequences, gene, labels in tqdm(train_loader):
            break
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            sequences, gene= sequences.to(device).unsqueeze(2), gene.to(device)
            
            optimizer.zero_grad()
            #outputs = model(images, sequences, gene)
            
            outputs = model(images, sequences, gene)
            # Calculate total loss by iterating over outputs and ground truths
            loss = sum(criterion(output.squeeze(), y.float()) for output, y in zip(outputs, labels))

            #loss = torch.sum(torch.tensor([criterion(output.squeeze(), y.float()) for output, y in zip(outputs, labels)]))
            #loss = criterion(outputs.squeeze(), labels.float())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples =0

        val_pred_probs, val_pred_labels, val_true_labels = [], [], []
        
        with torch.no_grad():
            for images, sequences, gene, labels in tqdm(val_loader):
                images = images.to(device)
                labels = [label.to(device) for label in labels]
                sequences, gene= sequences.to(device).unsqueeze(2), gene.to(device)

                outputs = model(images, sequences, gene)
                
                # Calculate total loss by iterating over outputs and ground truths
                #loss = torch.sum(criterion(output.squeeze(), y.float()) for output, y in zip(outputs, labels))
                loss = torch.sum(criterion(output.squeeze(), y.float()) for output, y in zip(outputs, labels))
                
                # loss = criterion(out1.squeeze(), y1) + criterion(out2, y2) + criterion(out3, y3)
                # loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item() * sequences.size(0)
                
                # Assuming threshold of 0.5 for binary classification
                predicted_labels = (outputs.squeeze() > 0.5).float()
                
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item() * sequences.size(0)
                
                correct_predictions += (predicted_labels == labels.float()).sum().item()
                total_samples += labels.size(0)

                val_pred_probs.extend(outputs.cpu().numpy())
                val_pred_labels.extend(predicted_labels.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/best_model_no_pretrain.pth')
            # log_model(experiment, model, model_name=f"CurModel_{epoch}")

        print(len(val_pred_labels))
        print(len(val_true_labels))

        val_acc, avg_train_loss, avg_val_loss = correct_predictions/len(val_loader.dataset), running_loss/len(train_loader.dataset), val_loss/len(val_loader.dataset)
        #val_true_labels, val_pred_probs = val_true_labels.cpu().numpy(), val_pred_probs.cpu().numpy()
        val_auc = roc_auc_score(val_true_labels, val_pred_probs)

        experiment.log_metrics({'Validation Accuracy': val_acc, 'Avg Training Loss': avg_train_loss, 'Avg Validation Loss': avg_val_loss, 'Validation AUC': val_auc}, epoch=epoch)
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_acc}, Validation AUC: {val_auc}")
        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_aucs.append(val_auc)
    
    plt.plot(val_losses, label="validation loss")
    plt.plot(train_losses, label="training loss")
    plt.legend()
    plt.axis('on')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{MODEL_SAVE_PATH}/loss_curve_no_pretrain.png')
    plt.clf()

    plt.plot(val_aucs, label="validation AUCs")
    plt.legend()
    plt.axis('on')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.savefig(f'{MODEL_SAVE_PATH}/auc_across_epochs_curve_no_pretrain.png')
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

from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map  # For progress bar with multiprocessing

# Define a function for generating and saving a single image
def save_curve_as_image(curve_idx):
    sequence = curve_dict[curve_idx][:40]
    imgs_folder = 'data/curve_img_human_fn'
    if not os.path.exists(imgs_folder):
        os.makedirs(imgs_folder, exist_ok=True)  # Ensure thread-safe directory creation
    plt.plot(sequence, linewidth=6)
    plt.axis('off')  # This will turn off the axis labels and ticks
    plt.savefig(f'{imgs_folder}/curve_{curve_idx}.png')
    plt.close()


class ImageSequenceDataset(Dataset):
    def __init__(self, curve_dict, target_df,img_directory = 'data/curve_imgs/', train=True, sequence_len=40, 
                 mean=0, std=1):
        self.curve_dict = curve_dict
        self.target_df = target_df
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

        #target data retrieval
        target = self.target_df.loc[self.target_df['curve_idx'] == curve_idx, 'groundtruth_target'].values[0]

        return curve_img, sequence_normalized, torch.tensor(target, dtype=torch.long)
    
class ImageModel(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(ImageModel, self).__init__()

        self.latent_dim = latent_dim
        
        # Image processing via EfficientNet_V2_L
        self.effnet = models.efficientnet_v2_l(pretrained=True)
        num_ftrs = self.effnet.classifier[1].in_features
        self.effnet.classifier = nn.Linear(num_ftrs, self.latent_dim)  # Adjusting to output a 512-dimensional 

        # FC Layers
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, sequence):
        # Image processing
        img_latent = self.effnet(image)
        output = self.fc(img_latent)
        return output
    

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.version.cuda)
print(torch.__version__)
print(device)


if __name__ == "__main__":
    ###########################################
    ## Load the data
    ###########################################

    # with open('data/groundtruth_df_curve_dict_split_v2.pkl', 'rb') as file:
    #     curve_dict = pkl.load(file)
    
    with open('data/human_label_curve_dict_fn.pkl', 'rb') as file: # 'data/karlen_curve_dict.pkl' new_full_curve_dict_fn_v1.pkl
        curve_dict = pkl.load(file)

    #target_df = pd.read_csv('data/groundtruth_df_target_data_split_v2.csv')
    target_df = pd.read_csv('data/human_label_df_target_data_split_v1.csv') # 'data/karlen_target_data.csv' new_groundtruth_df_target_data_v1.csv
    
    ###########################################
    ## Get the right normalization values
    ###########################################

    target_df_filtered = target_df[target_df['split']=='train']
    #target_df_filtered = target_df[target_df['split']=='test']
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

    #imgs_folder = 'data/curve_imgs_new_cleaner'
    curve_indices = list(curve_dict.keys())

    # Using ProcessPoolExecutor to parallelize the loop
    # Adjust `max_workers` as per your system's CPU resources if necessary
    with ProcessPoolExecutor(max_workers=os.cpu_count()//2) as executor:
        # Wrap with tqdm for a progress bar
        list(process_map(save_curve_as_image, curve_indices, chunksize=10, max_workers=os.cpu_count()//2))


    # for idx in tqdm(range(len(curve_dict.keys()))):
    #     curve_idx = list(curve_dict.keys())[idx]
    #     sequence = curve_dict[curve_idx][:40]

    #     if not os.path.exists(imgs_folder):
    #         os.makedirs(imgs_folder)
    #     plt.plot(sequence, linewidth=6)
    #     plt.axis('off')  # This will turn off the axis labels and ticks
    #     # plt.axis('on') 
    #     # plt.show()
    #     plt.savefig(f'{imgs_folder}/curve_{curve_idx}.png')
    #     plt.clf()
    
    
    # ###########################################
    # ## Set-up data objects
    # ###########################################

    # # Create Dataset and DataLoader
    # train_dataset = ImageSequenceDataset(curve_dict, target_df,img_directory = 'data/curve_imgs/', train=True, sequence_len=40,
    #                                      mean=norm_mean, std = norm_std)
    # val_dataset = ImageSequenceDataset(curve_dict, target_df,img_directory = 'data/curve_imgs/', train=False, sequence_len=40,
    #                                    mean=norm_mean, std = norm_std)
    
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


    # ###########################################
    # ## Initialize model
    # ###########################################

    # # Adjust the parameters as per your needs
    # sequence_length = 40  # Suppose the length of your sequence is 100
    # input_size = 1  # Number of input features per sequence element
    # hidden_size = 512
    # latent_dim = 512
    # num_layers = 3
    # num_epoch = 50

    # model = ImageModel(input_size, hidden_size, latent_dim)
    # model.to(device)  # If you are using GPU


    # ###########################################
    # ## Set up metric logging
    # ###########################################

    # experiment = Experiment(
    #     api_key="7smwpzl0FeZJcESqBITDniX7I",
    #     project_name="classification-arch-search",
    #     workspace="pcr-simulation"
    # )

    # experiment.set_name('ImageModel_MetricTest')


    # ###########################################
    # ## Training Loop
    # ###########################################

    # criterion = nn.BCELoss()  # Binary cross-entropy loss
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # best_val_loss = float('inf')

    # print('Device:', device)

    # for epoch in tqdm(range(num_epoch)):  # Choose the number of epochs
    #     model.train()
    #     running_loss = 0.0

    #     print('Starting epoch', epoch)
        
    #     for images, sequences, labels in tqdm(train_loader):
    #         images = images.to(device)
    #         sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)
            
    #         optimizer.zero_grad()
    #         outputs = model(images, sequences)
    #         loss = criterion(outputs.squeeze(), labels.float())

    #         loss.backward()
    #         optimizer.step()
            
    #         running_loss += loss.item() * sequences.size(0)
        
    #     # Validation
    #     model.eval()
    #     val_loss = 0.0
    #     correct_predictions = 0
    #     total_samples =0

    #     val_pred_probs, val_pred_labels, val_true_labels = [], [], []
        
    #     with torch.no_grad():
    #         for images, sequences, labels in tqdm(val_loader):
    #             images = images.to(device)
    #             sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)

    #             outputs = model(images, sequences)
    #             loss = criterion(outputs.squeeze(), labels.float())
    #             val_loss += loss.item() * sequences.size(0)

    #             # Assuming threshold of 0.5 for binary classification
    #             predicted_labels = (outputs.squeeze() > 0.5).float()
                
    #             loss = criterion(outputs.squeeze(), labels.float())
    #             val_loss += loss.item() * sequences.size(0)
                
    #             correct_predictions += (predicted_labels == labels.float()).sum().item()
    #             total_samples += labels.size(0)

    #             val_pred_probs.extend(outputs.cpu().numpy())
    #             val_pred_labels.extend(predicted_labels.cpu().numpy())
    #             val_true_labels.extend(labels.cpu().numpy())
                
    #     # Save model if it's the best so far
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         if not os.path.exists('output/image_model/'):
    #             os.makedirs('output/image_model/')
    #         torch.save(model.state_dict(), 'output/image_model/best_model_ep50.pth')
    #         log_model(experiment, model, model_name=f"CurModel_{epoch}")

    #     print(len(val_pred_labels))
    #     print(len(val_true_labels))

    #     val_acc, avg_train_loss, avg_val_loss = correct_predictions/len(val_loader.dataset), running_loss/len(train_loader.dataset), val_loss/len(val_loader.dataset)
    #     #val_true_labels, val_pred_probs = val_true_labels.cpu().numpy(), val_pred_probs.cpu().numpy()
    #     val_auc = roc_auc_score(val_true_labels, val_pred_probs)

    #     experiment.log_metrics({'Validation Accuracy': val_acc, 'Avg Training Loss': avg_train_loss, 'Avg Validation Loss': avg_val_loss, 'Validation AUC': val_auc}, epoch=epoch)
    #     print(f"Epoch {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_acc}, Validation AUC: {val_auc}")
        
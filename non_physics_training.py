import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import torch.nn.functional as F

class SequencesDataset(Dataset):
    def __init__(self, curve_dict, target_df):
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
        self.lstm = nn.LSTM(input_size=conv_out_channels*(seq_len-kernel_size+1), hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # swap sequence length and feature dimension for Conv1D
        x = F.relu(self.conv1(x))
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

    with open('data/groundtruth_df_curve_dict.pkl', 'rb') as file:
        curve_dict = pickle.load(file)
    target_df = pd.read_csv('data/groundtruth_df_target_data.csv')  # Load your DataFrame here

    # Create Dataset and DataLoader
    dataset = SequencesDataset(curve_dict, target_df)
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.25, random_state=42)

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

    ###########################################
    ## Training Loop
    ###########################################

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float('inf')

    for epoch in range(50):  # Choose the number of epochs
        model.train()
        running_loss = 0.0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)
            
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
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device).unsqueeze(2), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sequences.size(0)
                corrects += (outputs.argmax(1) == labels).sum().item()
                
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch}, Training Loss: {running_loss/len(train_loader.dataset)}, Validation Loss: {val_loss/len(val_loader.dataset)}, Validation Accuracy: {corrects/len(val_loader.dataset)}")


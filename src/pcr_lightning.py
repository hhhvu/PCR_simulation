import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from torchvision import models

class Classifier(pl.LightningModule):
    def __init__(self, num_classes=2, init_lr=1e-4):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        self.loss = nn.BCEWithLogitsLoss()

        self.accuracy = torchmetrics.Accuracy(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        self.training_outputs = []
        self.validation_outputs = []

    def get_xy(self, batch):
        x, y = batch[0], batch[1]
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        ## TODO: get predictions from your model and store them as y_hat
        y_hat = self.forward(*x)
        loss = self.loss(y_hat,y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(*x)

        print(y)
        print(y_hat)

        loss = self.loss(y_hat,y)

        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        
        self.log("train_auc", self.auc(y_hat, y), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        
        self.log("val_auc", self.auc(y_hat, y), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def configure_optimizers(self):
        ## TODO: Define your optimizer and learning rate scheduler here (hint: Adam is a good default)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr, betas = (0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor':'val_loss'}}

class FusionModel(Classifier):
    """
        Model that takes in sequence and image data and outputs single prediction head.
    """
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, init_lr=1e-4, pretrained=True):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        
        # Image processing via EfficientNet_V2_L
        # TODO change to true
        # self.effnet = models.efficientnet_v2_l(pretrained=True)
        # num_ftrs = self.effnet.classifier[1].in_features
        # self.effnet.classifier = nn.Linear(num_ftrs, self.latent_dim)  # Adjusting to output a 512-dimensional 

        if pretrained:
            self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            self.vit = models.vit_b_32(pretrained=False)
        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  # Adjusting to output a 512-dimensional 

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes
        neural_net_input = self.latent_dim*2

        # Fusion of image and sequence representations
        self.fc = nn.Sequential(
            nn.Linear(neural_net_input, 512),  # Concatenated vectors are of size 1024 (512 from image + 512 from sequence)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, sequence):
        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        fusion = torch.cat((img_latent, seq_latent), dim=1)
        output = self.fc(fusion)

        return output.squeeze()


class GeneFusionModel(Classifier):
    """
        Model that takes in sequence, image, and gene data and outputs multiple prediction heads.
    """
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=1, init_lr=1e-4):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

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
        self.lstm_delta = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state_delta = (torch.zeros(num_layers, sequence_length-1, hidden_size), torch.zeros(num_layers, sequence_length-1, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc_delta = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim*3 + genes

        # Fusion of image and sequence representations
        self.fc = nn.Sequential(
            nn.Linear(neural_net_input, 512),  # Concatenated vectors are of size 1024 (512 from image + 512 from sequence)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            )

        # Prediction heads
        #self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Calculating delta
        delta_seq = sequence[:, 1:] - sequence[:, :-1] #taking first difference
        lstm_out_delta, _ = self.lstm_delta(delta_seq)
        seq_latent_delta = self.lstm_fc_delta(lstm_out_delta[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), seq_latent_delta), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        #outputs = [torch.sigmoid(head(output)) for head in self.heads]

        return output.squeeze()


class GeneEnsembleModel(Classifier):
    """
        Model that takes in sequence, image, gene data, and igi call and outputs single prediction head. Note that the FusionModel loaded in must have 3 output heads.
    """
    def __init__(self, input_size, hidden_size, latent_dim, sequence_length, num_layers=5, init_lr=1e-4, fusion_path=None):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.fusion = FusionModel(input_size, hidden_size, latent_dim, sequence_length, num_layers=num_layers)
        if fusion_path != None:
            self.fusion.load_state_dict(torch.load(fusion_path))

        self.fc = nn.Linear(4, 1)

    def forward(self, image, sequence, genes, igi_call):
        x = self.fusion(image, sequence, genes)
        x = torch.cat(x + [igi_call.view(-1, 1)], dim=1)
        x = torch.sigmoid(self.fc(x))
        return x.squeeze()
    
    
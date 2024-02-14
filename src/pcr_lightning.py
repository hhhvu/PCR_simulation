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
        self.loss = nn.BCELoss()

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
        #y_hat = self.forward(x)
        loss = sum(self.loss(y_hat[:,i],y[:,i]) for i in range(3))

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(*x)
        #y_hat = self.forward(x)
        loss = sum(self.loss(y_hat[:,i],y[:,i]) for i in range(3))

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        y_hat = self.forward(*x)

        #loss = self.loss(y_hat,y)
        loss = sum(self.loss(y_hat[:,i],y[:,i]) for i in range(3))

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        
        self.log("train_auc", self.auc(y_hat, y), sync_dist=True, prog_bar=True)
        self.log("train_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        
        self.log("val_auc", self.auc(y_hat, y), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        #self.validation_outputs = []
        
        # save to process later for evaluation
        torch.save(y_hat, 'y_hat_val_image.pt')
        torch.save(y, 'y_val_true.pt')
    
    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)

        self.log("val_auc", self.auc(y_hat, y), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        self.test_outputs = []

        # save to process later for evaluation
        torch.save(y_hat, 'y_hat_test_image.pt')
        torch.save(y, 'y_test_true.pt')

    def configure_optimizers(self):
        ## TODO: Define your optimizer and learning rate scheduler here (hint: Adam is a good default)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor':'val_loss'}}
        return optimizer

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
        Model that takes in sequence, image, and gene data and outputs one prediction head.
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
        fusion = torch.cat((img_latent, seq_latent, genes, seq_latent_delta), dim=1)
        output = self.fc(fusion)

        return output.squeeze()

class SeqModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim

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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        fusion = seq_latent
        # fusion = torch.cat((seq_latent), dim=1)
        # fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), seq_latent_delta), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class SeqDeltaModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim + self.delta

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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Calculating delta
        delta_latent = torch.max(sequence, dim=1)[0] - torch.min(sequence, dim=1)[0]
        delta_latent = delta_latent.expand((-1, self.delta))

        # Fusion
        fusion = torch.cat((seq_latent, delta_latent), dim=1)

        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class SeqCurveModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4, pretrained=True):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta
        self.pretrained = pretrained
        
        if self.pretrained:
            self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            self.vit = models.vit_b_32(pretrained=False)

        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Fusion
        fusion = torch.cat((img_latent, seq_latent), dim=1)

        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class SeqDeltaGeneModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim + self.delta + genes

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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Calculating delta
        delta_latent = torch.max(sequence, dim=1)[0] - torch.min(sequence, dim=1)[0]
        delta_latent = delta_latent.expand((-1, self.delta))

        # Fusion
        fusion = torch.cat((seq_latent, genes.squeeze(1), delta_latent), dim=1)

        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class SeqGeneModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim + genes

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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Fusion
        # fusion = seq_latent
        fusion = torch.cat((seq_latent, genes.squeeze(1)), dim=1)
        # fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), seq_latent_delta), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class SeqCurveGeneModel(Classifier):
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4, pretrained=True):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta
        self.pretrained = pretrained
        
        if self.pretrained:
            self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            self.vit = models.vit_b_32(pretrained=False)

        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  

        # Sequence processing via LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_state = (torch.zeros(num_layers, sequence_length, hidden_size), torch.zeros(num_layers, sequence_length, hidden_size))
        # Final fully connected layer to ensure the LSTM output has a size of 512
        self.lstm_fc = nn.Linear(hidden_size, self.latent_dim)

        # Caluclate neural_net input size after appending genes and delta latent
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
            nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence, genes):
        # Sequence processing
        lstm_out, _ = self.lstm(sequence)
        seq_latent = self.lstm_fc(lstm_out[:, -1, :])  # Taking the last output from LSTM for the whole sequence

        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Fusion
        fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1)), dim=1)

        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class CurveShapeModel(Classifier):
    """
        Model that solely finetunes the ViT model from the sequence.
    """
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4, pretrained=True):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta
        self.pretrained = pretrained
        
        if self.pretrained:
            self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            self.vit = models.vit_b_32(pretrained=False)

        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  

        # Fusion of image and sequence representations
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),  # Concatenated vectors are of size 1024 (512 from image + 512 from sequence)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image):
        # Image processing
        img_latent = self.vit_classifier(self.vit(image))
        output = self.fc(img_latent)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

class CurveShapeDeltaModel(Classifier):
    """
        Model that solely finetunes the ViT model from the sequence.
    """
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4, pretrained=True):
        super().__init__(num_classes=2, init_lr=init_lr)
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.delta = delta
        self.pretrained = pretrained
        
        if self.pretrained:
            self.vit = models.vit_b_32(weights='IMAGENET1K_V1')
        else:
            self.vit = models.vit_b_32(pretrained=False)

        num_ftrs = self.vit.num_classes
        self.vit_classifier = nn.Linear(num_ftrs, self.latent_dim)  

        neural_net_input = self.latent_dim + self.delta

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
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            )

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_heads)])

    def forward(self, image, sequence):
        # Image processing
        img_latent = self.vit_classifier(self.vit(image))

        # Calculating delta
        delta_latent = torch.max(sequence, dim=1)[0] - torch.min(sequence, dim=1)[0]
        delta_latent = delta_latent.expand((-1, self.delta))

        # Fusion
        fusion = torch.cat((img_latent, delta_latent), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)

class GeneFusionHeadsModel(Classifier):
    """
        Model that takes in sequence, image, and gene data and outputs multiple prediction heads (for pred, igi_fp, igi_fn).
    """
    def __init__(self, input_size=1, hidden_size=512, latent_dim=512, sequence_length=40, num_layers=5, genes=6, delta=64, num_heads=3, init_lr=1e-4):
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

        # Caluclate neural_net input size after appending genes and delta latent
        neural_net_input = self.latent_dim*2 + genes + delta

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

        # Fusion
        fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), delta_latent), dim=1)
        # fusion = torch.cat((img_latent, seq_latent, genes.squeeze(1), seq_latent_delta), dim=1)
        output = self.fc(fusion)

        # Get predictions for each head
        outputs = torch.stack([torch.sigmoid(head(output)) for head in self.heads], dim=-1)

        return outputs.squeeze()
    
    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])

        self.log("val_auc", self.auc(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat[:,0], y[:,0]), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

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
    
    
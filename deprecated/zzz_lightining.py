import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from src.cindex import concordance_index

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=9, init_lr=1e-4):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes

        # Define loss fn for classifier
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.auc = torchmetrics.AUROC(task="binary" if self.num_classes == 2 else "multiclass", num_classes=self.num_classes)

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch):
        if isinstance(batch, list):
            x, y = batch[0], batch[1]
        else:
            assert isinstance(batch, dict)
            x, y = batch["x"], batch["y_seq"][:,0]
        return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)

        ## TODO: get predictions from your model and store them as y_hat
        y_hat = self.forward(x)
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

        #raise NotImplementedError("Not implemented yet")
        y_hat = self.forward(x)

        loss = self.loss(y_hat,y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.get_xy(batch)
        #raise NotImplementedError("Not implemented yet")
        y_hat = self.forward(x)

        loss = self.loss(y_hat,y)

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
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        if self.num_classes == 2:
            probs = F.softmax(y_hat, dim=-1)[:,-1]
        else:
            probs = F.softmax(y_hat, dim=-1)

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        ## TODO: Define your optimizer and learning rate scheduler here (hint: Adam is a good default)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr, betas = (0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor':'val_loss'}}


class MLP(Classifer):
    def __init__(self, input_dim=28*28*3, hidden_dim=128, num_layers=1, num_classes=9, use_bn=False, init_lr = 1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.use_bn = use_bn

        layers = []
        in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim  # Set input dim of next layer to hidden dim

        self.hiddens = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_classes)
        self.n_outputs = num_classes

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size,-1)
        x = self.hiddens(x)
        return self.output(x)

class Linear(Classifer):
    def __init__(self, input_dim=28*28*3,  num_classes=9, init_lr = 1e-3, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters()

        self.n_outputs = num_classes
        self.model = nn.Linear(input_dim, self.n_outputs)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size,-1)
        pred = self.model(x)
        return pred

class CNN(Classifer):
    def __init__(self, img_dim=28, in_channels = 3, n_filters=128, num_layers=1, num_classes=9, 
                 use_bn=False, init_lr = 1e-3, 
                 pool_every=2, conv_kernel_size =3,
                 pool_kernel_size =2, fc_hidden_size=128, **kwargs):
        
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 

        self.n_filters = n_filters
        self.use_bn = use_bn

        layers = []
        in_channels = 3  
        for i in range(num_layers):
            pad = (conv_kernel_size - 1) // 2 # pad such that we maintain the image dimension
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, self.n_filters, kernel_size=conv_kernel_size, stride=1, padding=pad),
                nn.ReLU(),
                )
            if self.use_bn:
                conv_layer.add_module('batch_norm', nn.BatchNorm2d(n_filters))
            layers.append(conv_layer)
            if i % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            
            in_channels = n_filters  # Update the input channels for the next layer

        self.layers = nn.Sequential(*layers)
        final_size = img_dim // (pool_kernel_size ** (num_layers//pool_every))

        if final_size == 0:
            raise ValueError("The final size of the output feature map is 0, which is invalid. Adjust the number of layers or the size of the input image.")

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * final_size * final_size, fc_hidden_size),  
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes),  
        )

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)

class ResNet(Classifer):
    def __init__(self, input_dim=28*28*3, num_classes=9, init_lr = 1e-3, pretrained=True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 

        self.network = torchvision.models.resnet18(pretrained=pretrained)
        self.network.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        return self.network(x)

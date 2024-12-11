import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score

class MLP(pl.LightningModule):
    
    def __init__(self, num_inputs=100, num_hidden_layers=2, hidden_dim=128, lr=5e-4, weight_decay=1e-4, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()

        linears = []
        input_dim = num_inputs
        for layer in range(num_hidden_layers):
            linears.append(nn.Linear(input_dim, hidden_dim))
            linears.append(nn.ReLU())
            input_dim = hidden_dim
        self.fc = nn.Sequential(*linears)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        x = self.fc(x)  # Pass through hidden layers
        logits = self.out(x)  # Output layer
        return logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr/50)
        return [optimizer], [lr_scheduler]
    
    def calc_f1(self, logits, y):
        """
        Calculate the F1 score.

        Parameters:
            logits (torch.Tensor): Predicted logits of shape (batch_size,).
            y (torch.Tensor): True labels of shape (batch_size,).

        Returns:
            float: F1 score.
        """
        preds = (logits > 0).float()  # Convert logits to binary predictions
        return f1_score(y.cpu(), preds.cpu(), average="binary")
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        self.log("train_loss", loss)
        logits_detached, y_detached = logits.detach(), y.detach()
        acc = ((logits_detached > 0.0).float() == y_detached).float().mean()
        self.log("train_acc", acc)
        f1 = self.calc_f1(logits_detached, y_detached)
        self.log("train_f1", f1)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1)
        logits = self(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        logits_detached, y_detached = logits.detach(), y.detach()
        acc = ((logits_detached > 0.0).float() == y_detached).float().mean()
        f1 = self.calc_f1(logits_detached, y_detached)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_f1", f1)
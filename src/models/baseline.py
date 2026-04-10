import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from src.models.cnn import TimmFeatureExtractor

class BaselineVideoClassifier(pl.LightningModule):
    def __init__(self, num_classes, num_frames):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_frames = num_frames
        
        self.feature_extractor = TimmFeatureExtractor()
        # Freeze CNN if you only want to train the classifier
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
        self.classifier = nn.Linear(self.feature_extractor.embed_dim * num_frames, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        B, F, C, H, W = x.size()
        # Reshape to push all frames through the CNN in one batch
        x = x.view(B * F, C, H, W)
        features = self.feature_extractor(x)
        
        # Reshape back to (B, F * embed_dim) and classify
        features = features.view(B, -1)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(logits, y), prog_bar=True)
        self.log('val_macro_f1', self.val_f1(logits, y), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
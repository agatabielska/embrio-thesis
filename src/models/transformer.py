import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
import torch.nn as nn
from src.models.cnn import TimmFeatureExtractor
import torch

class VideoTransformer(pl.LightningModule):
    def __init__(self, model_name='mobilenetv4_conv_medium.e500_r224_in1k', num_frames=16, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_extractor = TimmFeatureExtractor(model_name)
        embed_dim = self.feature_extractor.embed_dim
        print(f"Detected Feature Dimension: {embed_dim} {type(embed_dim)}")
        
        # TODO: fix this hack
        embed_dim = 1280
        
        # Transformer setup
        self.pos_encoding = nn.Parameter(torch.randn(1, num_frames, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 2-class output (Logits for Class 0 vs Class 1)
        self.classifier = nn.Linear(embed_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = BinaryAccuracy()
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        
        # Extract features
        features = self.feature_extractor(x) # (B*F, d)
        features = features.view(B, F, -1) # (B, F, d)
        
        print(f"Feature shape after extractor: {features.shape}")
        print(f"Positional encoding shape: {self.pos_encoding.shape}")
        
        # Temporal processing
        x = features + self.pos_encoding
        x = self.transformer(x)
        
        # Global average pool over frames
        x = x.mean(dim=1) 
        return self.classifier(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.log("train/loss", loss)
        self.log("train/acc", self.accuracy(preds, targets), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.accuracy(preds, targets), prog_bar=True)
        self.log("val/f1", self.f1(preds, targets))
        self.log("val/auroc", self.auroc(preds, targets))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
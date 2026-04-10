import timm
import torch
import torch.nn as nn

class TimmFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        # num_classes=0 removes the classification head and returns the global pool features
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the embedding dimension automatically
        self.embed_dim = self.model.num_features

    def forward(self, x):
        # x: (Batch * Frames, 3, 224, 224)
        return self.model(x) # Returns (Batch * Frames, embed_dim)
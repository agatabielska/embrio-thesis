import os
import cv2
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, F1Score
import math

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        """
        Expects a directory structure: root_dir/class_name/video.avi
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.avi'):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_path, label = self.samples[idx]
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample evenly spaced frames
        indices = torch.linspace(0, max(0, total_frames - 1), self.num_frames).long()
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i.item())
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Fallback to zero-tensor if frame reading fails
                frames.append(torch.zeros((224, 224, 3), dtype=torch.uint8).numpy())
                
        cap.release()
        
        # Convert list of frames to (num_frames, C, H, W)
        frames_tensor = []
        for f in frames:
            f = T.ToTensor()(f)
            if self.transform:
                f = self.transform(f)
            frames_tensor.append(f)
            
        return torch.stack(frames_tensor), label

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, num_frames=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # In a real scenario, you would split this into train/val folders
        self.dataset = VideoDataset(self.data_dir, num_frames=self.num_frames, transform=self.transform)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Strip the final classification layer to get raw embeddings
        self.extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.embed_dim = resnet.fc.in_features # 512 for resnet18

    def forward(self, x):
        # x shape: (B * num_frames, C, H, W)
        features = self.extractor(x)
        return features.view(features.size(0), -1)
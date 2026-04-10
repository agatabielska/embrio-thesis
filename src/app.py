import wandb
import pytorch_lightning as pl
from src.data import VideoDataModule
from src.models.transformer import VideoTransformer
from src.models.baseline import BaselineVideoClassifier
from pytorch_lightning.loggers import WandbLogger

def main():
    # 1. Initialize WandB
    wandb.init(project="video-frame-analysis", name="transformer-approach-v1")
    wandb_logger = WandbLogger()

    # 2. Setup Data
    # Assumes /data/class_A/*.avi and /data/class_B/*.avi
    data_module = VideoDataModule(
        data_dir="./data/PGT-A", 
        batch_size=4, # Keep small for video memory
        num_frames=16
    )

    # 3. Initialize Model
    model = VideoTransformer(num_frames=16, lr=1e-4)
    
    # Or for the baseline:
    # model = BaselineVideoClassifier(num_classes=2, num_frames=16)

    # 4. Trainer with WandB
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        precision="16-mixed", # Faster training on modern GPUs
        log_every_n_steps=5
    )

    trainer.fit(model, data_module)
    
    wandb.finish()

if __name__ == "__main__":
    main()
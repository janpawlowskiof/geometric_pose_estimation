from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision

from app.dataset.panoptic.flat_panoptic_clip import FlatPanopticClip


class FlatDataModule(pl.LightningDataModule):
    def __init__(self, train_clip_path: Path, batch_size: int = 32):
        super().__init__()
        self.train_clip_path = train_clip_path
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.get_dataset(), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.get_dataset(), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.get_dataset(), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.get_dataset(), batch_size=self.batch_size)

    def get_dataset(self):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        return FlatPanopticClip(self.train_clip_path, transforms)

from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


class ProjModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_joints = 19
        self.num_dims = 3
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_joints * self.num_dims)

    def forward(self, x: torch.Tensor, inv_p: torch.Tensor) -> torch.Tensor:
        """
        :param x: images of shape Nx3xHxW
        :param inv_p: inverse projection matrices of shape Nx4x3
        :return: back-projected predictions of shape Nx4x[num_joints]
        """
        y = self.model(x)
        n, c = y.shape
        y = y.reshape([n, self.num_dims, self.num_joints])
        return torch.einsum("nft,ntj->nfj", inv_p, y)

    def training_step(self, batch) -> STEP_OUTPUT:
        image = batch["image"]
        inv_p = batch["inv_p"]
        raise NotImplementedError()

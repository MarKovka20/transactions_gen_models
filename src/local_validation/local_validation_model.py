"""Module containtaining LocalValidationModel class."""
from typing import Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Metric, MetricCollection

from ptls.data_load.padded_batch import PaddedBatch

class LocalValidationModelBase(pl.LightningModule):
    """
    PytorchLightningModule for local validation of backbone (e.g. CoLES) model of transactions representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        pred_head: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metrics: MetricCollection,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize LocalValidationModel with pretrained backbone model and 2-layer linear prediction head.

        Args:
            backbone (nn.Module) - backbone model for transactions representations
            pred_head (nn.Module) - prediction head for target prediction
            loss (Callable) - the loss to optimize while training. Called with (preds, targets)
            metrics (MetricCollection) - collection of metrics to track in train, val, test steps
            freeze_backbone (bool) - whether to freeze backbone weights while training
            learning_rate (float) - learning rate for prediction head training
        """
        super().__init__()
        self.backbone = backbone

        if freeze_backbone:
            # freeze backbone model
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.track_running_stats = False
                    m.eval()
    
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.lr = learning_rate
        self.pred_head = pred_head
        self.loss = loss
        self.train_metrics = metrics.clone("Train")
        self.val_metrics = metrics.clone("Val")
        self.test_metrics = metrics.clone("Test")

    def forward(self, inputs: PaddedBatch) -> tuple[torch.Tensor]:
        """Do forward pass through the local validation model.

        Args:
            inputs (PaddedBatch) - inputs if ptls format (no sampling)

        Returns a tuple of:
            * torch.Tensor of predicted targets
            * torch.Tensor with mask corresponding to non-padded times
        """
        out = self.backbone(inputs)
        preds = self.pred_head(out)
        return preds
    
    def shared_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def training_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ):
        """Training step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)
        train_loss = self.loss(preds, target)
        self.train_metrics(preds, target)

        self.log("train_loss", train_loss, on_epoch=True)
        self.log_dict(self.train_metrics, on_epoch=True) # type: ignore

        return train_loss

    def validation_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ):
        """Validation step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)
        val_loss = self.loss(preds, target)
        self.val_metrics(preds, target)

        self.log("val_loss", val_loss)
        self.log_dict(self.val_metrics) # type: ignore

    def test_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], batch_idx: int
    ):
        """Test step of the LocalValidationModel."""
        preds, target = self.shared_step(batch, batch_idx)

        self.test_metrics(preds, target)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True) # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer for the LocalValidationModel."""
        opt = torch.optim.Adam(self.pred_head.parameters(), lr=self.lr)
        return opt

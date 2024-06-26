"""File with the main logic for the COTIC method."""

from typing import Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from ptls.frames.abs_module import ABSModule
from ptls.nn.seq_encoder.containers import SeqEncoderContainer


class Cotic(ABSModule):
    """Cotic module in ptls format."""

    def __init__(
        self,
        encoder: DictConfig,
        head: DictConfig,
        loss: DictConfig,
        metrics: DictConfig,
        optimizer_partial: DictConfig,
        lr_scheduler_partial: DictConfig,
        head_start: Optional[int] = None,
    ) -> None:
        """Init Cotic module.

        Args:
        ----
            encoder (DictConfig): config for continuous convolutional sequence encoder instantiation
            head (DictConfig): config custom prediction head for Cotic model instantiation
            loss (DictConfig): config for module with Cotic losses instantiation
            metrics (DictConfig): config for module with Cotic metrics instantiation
            optimizer_partial (DictConfig): config for optimizer instantiation (ptls format)
            lr_scheduler_partial (DictConfig): config for lr scheduler instantiation (ptls format)
            head_start (int): if not None, start training prediction head after this epoch.
        """
        self.save_hyperparameters()
        enc: SeqEncoderContainer = instantiate(encoder)

        super().__init__(
            seq_encoder=enc,
            loss=instantiate(loss),
            optimizer_partial=instantiate(optimizer_partial, _partial_=True),
            lr_scheduler_partial=instantiate(lr_scheduler_partial, _partial_=True),
        )

        self.encoder = enc

        self.train_metrics = instantiate(metrics)
        self.val_metrics = instantiate(metrics)
        self.test_metrics = instantiate(metrics)

        self._head = instantiate(head)

        self.head_start = head_start

    def shared_step(
        self, batch: tuple[PaddedBatch, torch.Tensor]
    ) -> tuple[
        tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Shared training/validation/testing step.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)

        Retruns a tuple of:
            inputs - inputs for CCNN model: (event_times, event_types), for loss & metric computation
            outputs - outputs of the model: (encoded_outputs, (pred_times, pred_types))
        """
        encoded_output = self(
            batch[0]
        )  # out of CoticSeqEncoder (aka 'encoded_output' in Cotic)
        pred_times, pred_types = self._head(encoded_output.detach())

        inputs = self.seq_encoder._extract_times_and_features(  # type: ignore
            batch[0]
        )  # format is (event_times, event_types)
        outputs = (
            encoded_output,
            (
                pred_times,
                pred_types,
            ),
        )  # format is (encoded_output, (pred_times, pred_types))

        ll_loss, type_loss, time_loss = self._loss.compute_loss(  # type: ignore
            model=self.seq_encoder.seq_encoder.feature_extractor,  # type: ignore
            inputs=inputs,
            outputs=outputs,
        )

        return (inputs, outputs), (ll_loss, type_loss, time_loss)

    def training_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], _
    ) -> dict[str, torch.Tensor]:
        """Training step of the module.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)

        Returns:
        -------
            dict with train loss
        """
        (inputs, outputs), (ll_loss, type_loss, time_loss) = self.shared_step(batch)

        self.log("train_ll_loss", ll_loss, prog_bar=True)

        if self.head_start is not None and self.current_epoch >= self.head_start:
            self.log("train_type_loss", type_loss, prog_bar=True)
            self.log("train_time_loss", time_loss, prog_bar=True)

            self.train_metrics.update(inputs, outputs)

            return {"loss": ll_loss + type_loss + time_loss}

        return {"loss": ll_loss}

    def validation_step(
        self, batch: tuple[PaddedBatch, torch.Tensor], _
    ) -> dict[str, torch.Tensor]:
        """Training step of the module.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)

        Returns:
        -------
            dict with val loss
        """
        (inputs, outputs), (ll_loss, type_loss, time_loss) = self.shared_step(batch)

        self.log("val_ll_loss", ll_loss, prog_bar=True)

        if self.head_start is not None and self.current_epoch >= self.head_start:
            self.log("val_type_loss", type_loss, prog_bar=True)
            self.log("val_time_loss", time_loss, prog_bar=True)

            self.val_metrics.update(inputs, outputs)

            return {"loss": ll_loss + type_loss + time_loss}

        return {"loss": ll_loss}

    def test_step(self, batch: tuple[PaddedBatch, torch.Tensor], _) -> None:
        """Test step of the module.

        Args:
        ----
            batch (tuple[PaddedBatch, torch.Tensor]): padded batch that is fed into CoticSeqEncoder and labels (irrelevant here)
        """
        if self.head_start is not None:
            (inputs, outputs), _ = self.shared_step(batch)

            self.test_metrics.update(inputs, outputs)

    def training_epoch_end(self, _) -> None:
        """Compute and log metrics for a training epoch."""
        if self.head_start is not None and self.current_epoch >= self.head_start:
            return_time_metric, event_type_metric = self.train_metrics.compute()

            self.log("val_return_time_metric", return_time_metric, prog_bar=True)
            self.log("val_event_type_metric", event_type_metric, prog_bar=True)

    def validation_epoch_end(self, _) -> None:
        """Compute and log metrics for a validation epoch."""
        if self.head_start is not None and self.current_epoch >= self.head_start:
            return_time_metric, event_type_metric = self.val_metrics.compute()

            self.log("val_return_time_metric", return_time_metric, prog_bar=True)
            self.log("val_event_type_metric", event_type_metric, prog_bar=True)

    def test_epoch_end(self, _) -> None:
        """Compute and log metrics for a test epoch."""
        return_time_metric, event_type_metric = self.test_metrics.compute()

        self.log("test_return_time_metric", return_time_metric, prog_bar=True)
        self.log("test_event_type_metric", event_type_metric, prog_bar=True)

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        """Overwrite method as to fix bug in our PyTorch version."""
        scheduler.step(epoch=self.current_epoch)

    @property
    def is_requires_reduced_sequence(self):
        """COTIC does not reduce sequence by default."""
        return False

    @property
    def metric_name(self):
        """Validation monitoring metric name."""
        return "val_ll_loss"

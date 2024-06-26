"""Main coles learning script."""

from pathlib import Path
from typing import Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptls.frames import PtlsDataModule
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import Dataset

from src.modules import Cotic, CustomCoLES, TS2Vec, VanillaAE
from src.utils.create_trainer import create_trainer
from src.utils.logging_utils import get_logger

logger = get_logger(name=__name__)


def learn(
    data: tuple[list[dict], list[dict], list[dict]],
    backbone_cfg: DictConfig,
    logger_cfg: DictConfig,
    encoder_save_name: str,
) -> None:
    """Full pipeline for model fitting.

    Args:
    ----
        data (tuple[list[dict], list[dict], list[dict]]):
            train, val and test sets
        backbone_cfg (DictConfig): config with the following fields:
            dataset:
                Dataset config, additionally passed two keyword arguments:
                 - data (list of dicts) - the data in FeatureDict format (returned by preprocess)
                 - deterministic (bool) - true if randomness needs to be off, for test/val.
                (specified in 'config/dataset')
            module:
                LightningModule config, with train, test and val steps (specified in 'config/module')
            encoder:
                Config for encoder.
        logger_cfg (DictConfig): config with logger
        encoder_save_name (str): where to save encoder state dict.
    """
    train, val, test = data

    logger.info("Preparing datasets and datamodule")
    # Define our ColesDataset wrapper from the config
    train_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=train, deterministic=False
    )
    val_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=val, deterministic=True
    )
    test_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=test, deterministic=True
    )

    # Pytorch-lifestream datamodule for the model training and evaluation
    datamodule: PtlsDataModule = instantiate(
        backbone_cfg["datamodule"],
        train_data=train_data,
        valid_data=val_data,
        test_data=test_data,
    )

    # Instantiate the encoder (and, optionally, the decoder)
    module_args = {}
    module_args["encoder"] = backbone_cfg["encoder"]
    if "decoder" in backbone_cfg:
        module_args["decoder"] = backbone_cfg["decoder"]

    # Instantiate the LightningModule.
    # _recursive_=False to save all hyperparameters
    # as DictConfigs, to enable hp loading from lightning checkpoint
    module: Union[CustomCoLES, VanillaAE, Cotic, TS2Vec] = instantiate(
        backbone_cfg["module"], **module_args, _recursive_=False
    )

    trainer = create_trainer(
        logger=logger_cfg,
        metric_name=module.metric_name,
        **backbone_cfg["trainer"],
    )

    # Training the model
    trainer.fit(module, datamodule)

    # Load the checkpoint & recalculate val metrics
    # (if checkpointing is enabled)
    if not trainer.fast_dev_run and trainer.checkpoint_callback:
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        module.load_state_dict(checkpoint["state_dict"])

    trainer.validate(module, datamodule)

    # Optionally run test
    if is_overridden("test_step", module):
        trainer.test(module, datamodule)

    # Save the state_dict of the best sequence encoder
    saved_models_path = Path("saved_models")
    saved_models_path.mkdir(exist_ok=True)
    if not trainer.fast_dev_run:
        torch.save(
            module.encoder.state_dict(), saved_models_path / f"{encoder_save_name}.pth"
        )

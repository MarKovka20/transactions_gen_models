"""Main pooling learning script."""

from pathlib import Path

import numpy as np
import pandas as pd

import torch

from hydra import initialize, compose
from hydra.utils import instantiate, call

from pytorch_lightning import Trainer, seed_everything

from src.local_validation.local_validation_model import LocalValidationModelBase
from src.preprocessing import preprocess

from ptls.frames import PtlsDataModule

from src.datasets.coles import CustomColesDataset
from src.modules.coles import CustomCoLES

from src.pooling import PoolingModel

from torch.utils.data import Dataset
from src.utils.create_trainer import create_trainer
from typing import Union
from src.modules import Cotic, CustomCoLES, TS2Vec, VanillaAE
from pytorch_lightning.utilities.model_helpers import is_overridden
import torch.nn as nn


def learn_pooling(data, pooling_model = None, learn_config_name = "master", seed = 42, validate=False): 
    # config and seed
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name=learn_config_name)

    backbone_cfg=cfg["backbone"]
    seed_everything(seed)

    # data
    print("Data preprocessing...")
    train, val, test = data

    train_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=train, deterministic=False
    )
    val_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=val, deterministic=True
    )
    test_data: Dataset = instantiate(
        backbone_cfg["dataset"], data=test, deterministic=True
    )

    if not validate:
        val_data = None
    datamodule: PtlsDataModule = instantiate(
        backbone_cfg["datamodule"],
        train_data=train_data,
        valid_data=val_data,
        test_data=test_data,
    )

    print("Module creating...")
    # module (default)
    module_args = {}
    module_args["encoder"] = backbone_cfg["encoder"]
    if pooling_model is not None:
        backbone_cfg["encoder"]["hidden_size"] = pooling_model.embedding_size
    if "decoder" in backbone_cfg:
        print(backbone_cfg["decoder"])
        module_args["decoder"] = backbone_cfg["decoder"]

    module: Union[CustomCoLES, VanillaAE, Cotic, TS2Vec] = instantiate(
        backbone_cfg["module"], **module_args, _recursive_=False
    )

    # module (change)
    if pooling_model is not None:
        pooling_model.backbone.is_reduce_sequence = module.encoder.is_reduce_sequence
        module.encoder = pooling_model

        print("Freezing of weights...")
        # module (freeze)
        for m in module.encoder.backbone.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = False
                m.eval()

        for param in module.encoder.backbone.parameters():
            param.requires_grad = False

        for name, param in module.encoder.named_parameters():
            print(name, param.requires_grad)

    print("Trainer creating...")
    # trainer
    trainer = create_trainer(
        logger=cfg["logger"],
        metric_name=module.metric_name,
        **backbone_cfg["trainer"],
    )

    print("Training...")
    # training 
    try:
        trainer.fit(module, datamodule)
        trainer.validate(module, datamodule)
        if is_overridden("test_step", module):
            trainer.test(module, datamodule)
    except Exception as es:
        print(es)

    return module
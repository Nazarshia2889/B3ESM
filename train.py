import argparse
import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from tqdm import tqdm

from config import get_config
from data import build_loader
from models import build_model

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE: {device}")
assert "cuda" in str(device), "Use GPU, dum dum!"

def parse_option():
    parser = argparse.ArgumentParser("B3ESM training and evaluation script", add_help=False)
    parser = argparse.ArgumentParser("B3ESM training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--run_name", type=str, required=True, help="run name in W&B")
    parser.add_argument("--project", type=str, default="B3ESM", help="Name of WandB project")
    args = parser.parse_args()

    config = get_config(args)
    os.makedirs(config.OUTPUT, exist_ok=True)
    print(f"OUTPUT: {config.OUTPUT}")

    with open(os.path.join(config.OUTPUT, "config.yaml"), "w") as f:
        f.write(config.dump())

    return args, config

def train_simclr(args, config):
    trainer = pl.Trainer(default_root_dir=config.OUTPUT,
                         accelerator="gpu", #if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=config.TRAIN.EPOCHS,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor=config.TRAIN.MONITOR),
                                    LearningRateMonitor('epoch')],
                         logger=WandbLogger(name=args.run_name, project=args.project),
                         log_every_n_steps=10)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    num_inputs, dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    # pl.seed_everything(42)
    model = build_model(config, num_inputs)
    trainer.fit(model, data_loader_train, data_loader_val)
    print(f"Best model path: {trainer.checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    args, config = parse_option()
    train_simclr(args, config)
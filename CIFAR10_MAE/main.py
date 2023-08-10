import os
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchmetrics import Accuracy

import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt

from datamodule import Cifar10DataModule
from model import MAE
        
def train_model(**kwargs):
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=50,
        logger=CSVLogger(save_dir="logs/"),
        limit_val_batches=0.0,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )
    L.seed_everything(42)
    model = MAE(enc_depth = 6, dec_depth = 4)
    dm = Cifar10DataModule()
    
    trainer.fit(model=model, datamodule=dm)
    trainer.save_checkpoint("MAE.ckpt")
    # model = MAE.load_from_checkpoint("MAE.ckpt")

    # test_result = trainer.test(model=model, dataloaders=dm.test_dataloader())
    # result = {"test": test_result[0]["test_acc"]}

    return model

if __name__=='__main__':

    L.seed_everything(42)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    model = train_model()
    
    print("MAE train done")
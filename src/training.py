import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import pandas as pd
from config import PATIENCE, PATH_SUPERVISED_MODELS
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from time import time
from models import SimCLR

def train_simclr_model(model, train_loader, val_loader, max_epochs=100, lr=1e-3):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, monitor='val_loss', mode='min'),
            LearningRateMonitor(logging_interval='epoch')
        ]
    )
    trainer.fit(model, train_loader, val_loader)


def train_supervised_model(lightning_model, 
                           labeled_datamodule,
                           total_epochs = 200, 
                           model_name = "model_name"):
    t0 = time()
    # Define Callbacks
    callback_check = ModelCheckpoint(
        dirpath=PATH_SUPERVISED_MODELS, 
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="best-checkpoint",
        save_weights_only=True,
        verbose=True
    )
    callback_tqdm = RichProgressBar(leave=True)
    callback_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )
    # Initialize loggers
    csv_logger = CSVLogger(save_dir=PATH_SUPERVISED_MODELS, name=model_name)

    # Define Trainger
    trainer = Trainer(
        max_epochs=total_epochs,
        callbacks=[callback_check, callback_tqdm, callback_early_stop],
        # accelerator="auto",
        # devices="auto",
        accelerator="gpu",  
        devices=[1],        # Specify GPU 1 (CUDA device 1)
        logger=[csv_logger]
    )

    # Start Training
    trainer.fit(model=lightning_model, datamodule=labeled_datamodule)

    # Get best model
    best_model_path = callback_check.best_model_path

    # Save Results
    res = trainer.validate(ckpt_path=best_model_path, datamodule=labeled_datamodule)  
    df_metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    # Measure computing time
    t1 = time()
    computing_time_minutes = (t1-t0)/60
    # Return results
    return res, df_metrics, best_model_path, computing_time_minutes


def train_simclr(batch_size, train_loader, val_loader, max_epochs=500, patience=20, **kwargs):
    """
    Train a SimCLR model with early stopping.
    
    Args:
        batch_size (int): Batch size for training.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs to wait for improvement before stopping.
        **kwargs: Additional arguments for the SimCLR model.
        
    Returns:
        SimCLR model: The trained model loaded from the best checkpoint.
    """
    # Initialize loggers
    csv_logger = CSVLogger(os.path.join(PATH_SUPERVISED_MODELS, 'simclr_logs'), 
                           name="train_simclr_resnet")
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top1'),
        LearningRateMonitor('epoch'),
        EarlyStopping(monitor='val_acc_top1', patience=patience, mode='max')
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(PATH_SUPERVISED_MODELS, 'SimCLR'),
        accelerator='cuda',
        devices=[0],
        #num_nodes=2,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[csv_logger],
        callbacks=callbacks
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Set seed for reproducibility
    pl.seed_everything(0)
    
    # Initialize model
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Load the best checkpoint
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model








def train_supervised_model_v2(lightning_model, 
                           labeled_datamodule,
                           total_epochs = 200, 
                           model_name = "model_name"):
    t0 = time()
    # Define Callbacks
    callback_check = ModelCheckpoint(
        dirpath=PATH_SUPERVISED_MODELS, 
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="best-checkpoint",
        save_weights_only=True,
        verbose=True
    )
    callback_tqdm = RichProgressBar(leave=True)
    callback_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )
    # Initialize loggers
    csv_logger = CSVLogger(save_dir=PATH_SUPERVISED_MODELS, name=model_name)

    # Define Trainger
    trainer = Trainer(
        max_epochs=total_epochs,
        callbacks=[callback_check, callback_tqdm, callback_early_stop],
        # accelerator="auto",
        # devices="auto",
        accelerator="gpu",  
        devices=[1],        # Specify GPU 1 (CUDA device 1)
        logger=[csv_logger]
    )

    # Start Training
    trainer.fit(model=lightning_model, datamodule=labeled_datamodule)

    # Get best model
    best_model_path = callback_check.best_model_path

    # Save Results
    res = trainer.validate(ckpt_path=best_model_path, datamodule=labeled_datamodule)  
    df_metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    # Measure computing time
    t1 = time()
    computing_time_minutes = (t1-t0)/60
    # Return results
    return res, trainer
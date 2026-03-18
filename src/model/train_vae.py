## Importing necessary libraries:
import os

## Data manipulation
import numpy as np
import pandas as pd

## Metric
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

## Deep learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from torch import nn

from models.VAE import MyVAE, Frauddataset, vae_loss_function

from utils import load_params, setup_logger

logger = setup_logger("trainning_VAE", "VAE_error.log")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug("Device ready: %s", device)


def load_data_dataloader(file_path_feature: str,file_path_label:str, batch_size: int, shuffle:bool)-> DataLoader:
    '''Load numpy matrix into the dataloader to hand it to the model'''
    try:
        dataset = Frauddataset(featuresfile = file_path_feature, labelfile = file_path_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        logger.debug("data loaded into the dataloader with shape: %s", (len(dataset),{dataset.x.shape[1]}))
        return dataloader
    except Exception as e:
        logger.error("dataloading failed: %s", e)
        raise

def train_1epoch(model: MyVAE, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, beta: float, device: torch.device)-> float:
    '''Training one epoch'''
    try:
        model.train()
    
        train_loss = 0
        size = len(train_dataloader.dataset)
    
        for batch, x in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(x)
            loss = vae_loss_function(reconstructed_x, x, mu, logvar, beta)
        
       
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        Avg_loss = train_loss / size
        return Avg_loss
    except Exception as e:
        raise

def AUROC_AP_1epcoh(model, test_dataloader, device):
    '''Calculate AUROC and AP after one epoch'''
    model.eval()
    reconstruction_errors = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            reconstructed, mu, logvar = model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    reconstruction_errors = np.array(reconstruction_errors)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, reconstruction_errors)
    ap = average_precision_score(all_labels, reconstruction_errors)
    
    return auroc, ap





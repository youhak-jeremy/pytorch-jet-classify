from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, average_precision_score, auc, accuracy_score
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import os.path as path

def l1_regularizer(model, lambda_l1=0.01):
    #  after hours of searching, this man is a god: https://stackoverflow.com/questions/58172188/
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

def train(model, optimizer, loss, train_loader, L1_factor=0.0001, l1reg=True, device='cpu'):
    model.to(device)
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader, 0):
        x, y = data #data_batch
        # x = (B, F)
        x, y = x.float().to(device), y.float().to(device)
        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(x)
        criterion_loss = loss(outputs, y)
        if l1reg:
            reg_loss = l1_regularizer(model, lambda_l1=L1_factor)
        else:
            reg_loss = 0
        total_loss = criterion_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        step_loss = total_loss.item()
        train_loss += step_loss * x.size(0)
    train_loss /= len(train_loader)
    return model, train_loss


def val(model, loss, val_loader, L1_factor=0.01, device='cpu'):
    val_roc_auc_scores_list = []
    val_avg_precision_list = []
    val_losses = 0
    model.to(device)
    model.eval()
    with torch.set_grad_enabled(False):
        for i, data in enumerate(val_loader, 0):
            x, y = data
            x, y = x.float().to(device), y.float().to(device)
            outputs = model(x)
            criterion_loss = loss(outputs, y)
            reg_loss = l1_regularizer(model, lambda_l1=L1_factor)
            val_loss = criterion_loss + reg_loss
            x, y = x.cpu(), y.cpu()
            outputs = outputs.cpu()
            val_roc_auc_scores_list.append(roc_auc_score(np.nan_to_num(y.numpy()), np.nan_to_num(outputs.numpy())))
            val_avg_precision_list.append(average_precision_score(np.nan_to_num(y.numpy()), np.nan_to_num(outputs.numpy())))
            step_loss = val_loss.item()
            val_losses += step_loss * x.size(0)
    val_losses /= len(val_loader)
    return val_losses, np.average(val_avg_precision_list), np.average(val_roc_auc_scores_list)


def test(model, test_loader, device='cpu'):
    #device = torch.device('cpu') #required if doing a untrained init check
    accuracy_score_value_list = []
    roc_auc_score_list = []
    model.to(device)
    model.eval()
    with torch.no_grad():  # Evaulate pruned model performance
        for i, data in enumerate(test_loader):
            x, y = data
            x, y = x.float().to(device), y.float().to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            _, true_preds = torch.max(y, 1)
            accuracy_score_value_list.append(accuracy_score(np.nan_to_num(true_preds.cpu().numpy()), np.nan_to_num(preds.cpu().numpy())))
            roc_auc_score_list.append(roc_auc_score(np.nan_to_num(y.cpu().numpy()), np.nan_to_num(outputs.cpu().numpy())))
    return np.average(accuracy_score_value_list), np.average(roc_auc_score_list)
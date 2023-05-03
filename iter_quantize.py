# Import misc packages
import math
import json
import os
import os.path as path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from optparse import OptionParser
from typing import Tuple

# Import torch stuff
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

# Import our own code
import models
import jet_dataset
from training.early_stopping import EarlyStopping
from training.train_funcs import train, val, test
from tools.aiq import calc_AiQ 
from training.training_plots import plot_total_loss, plot_total_eff, plot_metric_vs_bitparam, plot_kernels
from tools.param_count import countNonZeroWeights
from tools.parse_yaml_config import parse_config
import tqdm
import time

import json

from constants import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

def store_model(model, save_path):
    # QUESTION: WILL THIS STORE QUANTIZATION PARAMS
    torch.save(model.state_dict(), save_path)

def load_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        return json.load(f)

def store_metrics(metrics, metrics_file):
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

def load_jet_data(train_file, test_file, yamlConfig):
    # Setup and split dataset
    full_dataset = jet_dataset.ParticleJetDataset(train_file, yamlConfig)

    train_size = int(TRAIN_SPLIT * len(full_dataset))  # 25% for Validation set, 75% for train set
    val_size = len(full_dataset) - train_size
    num_val_batches = math.ceil(val_size/BATCH_SIZE)
    num_train_batches = math.ceil(train_size/BATCH_SIZE)
    print("train_batches " + str(num_train_batches))
    print("val_batches " + str(num_val_batches))

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,[train_size,val_size])
        
    test_dataset = jet_dataset.ParticleJetDataset(test_file, yamlConfig)
    test_size = len(test_dataset)

    print("train dataset size: " + str(len(train_dataset)))
    print("validation dataset size: " + str(len(val_dataset)))
    print("test dataset size: " + str(len(test_dataset)))
    
    # Setup dataloaders with our dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8, pin_memory=True)  # FFS, have to use numworkers = 0 because apparently h5 objects can't be pickled, https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/issues/69

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size,
                                              shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_model(layer_quantization_specs):
    model = models.three_layer_model_custom_quant(layer_quantization_specs)
    return model

def train_loop(model, dataloaders, num_epochs, device):
    train_loader, val_loader, test_loader = dataloaders

    metrics = {}

    epoch_counter = 0
    # early_stopping = EarlyStopping(patience=ER_PATIENCE, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    batnorm = False

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_counter += 1
        # Train
        model, train_loss = train(model, optimizer, criterion, train_loader, L1_factor=L1FACTOR, l1reg=L1REG, device=device)

        # Validate
        val_loss, val_avg_precision, val_roc_auc_score = val(model, criterion, val_loader, L1_factor=L1FACTOR, device=device)

        metrics.setdefault('avg_train_losses', []).append(train_loss)
        metrics.setdefault('avg_valid_losses', []).append(val_loss)
        metrics.setdefault('avg_precision_scores', []).append(val_avg_precision)

        # Print epoch statistics
#         print('[epoch %d] train batch loss: %.7f' % (epoch + 1, train_loss))
#         print('[epoch %d] val batch loss: %.7f' % (epoch + 1, val_loss))
#         print('[epoch %d] val ROC AUC Score: %.7f' % (epoch + 1, val_roc_auc_score))
#         print('[epoch %d] val Avg Precision Score: %.7f' % (epoch + 1, val_avg_precision))
        # Check if we need to early stop
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     estop = True
        #     break
        # Test Unpruned, Base Quant model

    print("Base Quant Model: ")

    base_quant_params,_,_,_ = countNonZeroWeights(model)
    base_quant_accuracy_score, base_quant_roc_score = test(model, test_loader)
    aiq_dict, aiq_time = calc_AiQ(model,test_loader,batnorm=False,device=device)


    metrics['base_quant_params'] = base_quant_params
    metrics['base_quant_accuracy_score'] = base_quant_accuracy_score
    metrics['base_quant_roc_score'] = base_quant_roc_score

    metrics['performance'] = metrics['base_quant_accuracy_score']
    metrics['efficiency'] = aiq_dict['net_efficiency']
    return metrics


def execute(
        train_file,
        test_file,
        yamlConfig,
        output_dir,
        quantization_spec,
        num_epochs
    ):
    quant_spec_str = '_'.join(str(v) for v in quantization_spec)
    ckpt_save_path = f"checkpoints/model_{quant_spec_str}.ckpt"
    metrics_save_path = f"metrics/metrics_{quant_spec_str}.json"
    os.makedirs("checkpoints/", exist_ok = True)
    os.makedirs("metrics/", exist_ok = True)

    if os.path.exists(ckpt_save_path):
        metrics = load_metrics(metrics_save_path)
        return metrics["performance"], metrics["efficiency"]
    
    # create given output directory if it doesnt exist
    if not path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # quantize model according to model_set
    model = get_model(quantization_spec)

    # Setup cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        print("cuda:0 device type: {}".format(torch.cuda.get_device_name(0)))
    else:
        use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        device = torch.device("mps") if use_mps else device
    
    print("Using Device: {}".format(device))

    dataloaders = load_jet_data(train_file, test_file, yamlConfig)
    metrics = train_loop(model, dataloaders, num_epochs, device)
    store_model(model, ckpt_save_path)
    store_metrics(metrics, metrics_save_path)

    # Time for filenames
    filename = path.join(output_dir, f'weight_dist_{quant_spec_str}.png')
    plot_kernels(model, text=' (Unpruned Quant Model)', output=filename)

    return metrics['performance'], metrics['efficiency']


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input_file'   ,action='store',type='string',dest='input_file'   ,default='', help='location of data to train off of')
    parser.add_option('-t','--test_file'   ,action='store',type='string',dest='test_file' ,default='', help='Location of test data set')
    parser.add_option('-o','--output_dir'   ,action='store',type='string',dest='output_dir' ,default='train_simple/', help='output directory')
    parser.add_option('-e','--num_epochs'   ,action='store',type='int', dest='epochs', default=100, help='number of epochs to train for')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-m', '--quantization_spec', type='str', dest='quantization_spec', default='32,12,8,6,4', help='comma separated list of which bit widths to run')
    (options,args) = parser.parse_args()
    yamlConfig = parse_config(options.config)

    execute(
        options.train_file,
        options.test_file,
        yamlConfig,
        options.output_dir,
        options.quantization_spec,
        options.num_epochs
    )

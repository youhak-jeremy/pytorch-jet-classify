import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import models
import jet_dataset
import h5py
import matplotlib.pyplot as plt
from optparse import OptionParser
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, average_precision_score, precision_recall_curve, auc
import torch.optim as optim
import yaml
from torchsummaryX import summary
import math
import seaborn as sn
from datetime import datetime
from itertools import cycle

def get_features(options, yamlConfig):
    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    print(treeArray.shape)
    print(treeArray.dtype.names)

    # List of features to use
    features = yamlConfig['Inputs']

    # List of labels to use
    labels = yamlConfig['Labels']

    # Convert to dataframe
    features_labels_df = pd.DataFrame(treeArray, columns=list(set(features + labels)))
    features_labels_df = features_labels_df.drop_duplicates()

    features_df = features_labels_df[features]
    labels_df = features_labels_df[labels]

    if 'Conv' in yamlConfig['InputType']:
        labels_df = labels_df.drop_duplicates()

    # Convert to numpy array
    features_val = features_df.values
    labels_val = labels_df.values

    if 'j_index' in features:
        features_val = features_val[:, :-1]  # drop the j_index feature
    if 'j_index' in labels:
        labels_val = labels_val[:, :-1]  # drop the j_index label
        print(labels_val.shape)

    if yamlConfig['InputType'] == 'Conv1D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['MaxParticles'], len(features) - 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index'] == labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values
            # features_val_i = features_val[index_values[0]:index_values[-1]+1,:-1] # drop the last feature j_index
            features_val_i = features_val[np.array(index_values), :]
            nParticles = len(features_val_i)
            # print("before", features_val_i[:,0])
            features_val_i = features_val_i[
                features_val_i[:, 0].argsort()[::-1]]  # sort descending by first value (ptrel, usually)
            # print("after", features_val_i[:,0])
            if nParticles > yamlConfig['MaxParticles']:
                features_val_i = features_val_i[0:yamlConfig['MaxParticles'], :]
            else:
                features_val_i = np.concatenate(
                    [features_val_i, np.zeros((yamlConfig['MaxParticles'] - nParticles, len(features) - 1))])
            features_2dval[i, :, :] = features_val_i

        features_val = features_2dval

    elif yamlConfig['InputType'] == 'Conv2D':
        features_2dval = np.zeros((len(labels_df), yamlConfig['BinsX'], yamlConfig['BinsY'], 1))
        for i in range(0, len(labels_df)):
            features_df_i = features_df[features_df['j_index'] == labels_df['j_index'].iloc[i]]
            index_values = features_df_i.index.values

            xbins = np.linspace(yamlConfig['MinX'], yamlConfig['MaxX'], yamlConfig['BinsX'] + 1)
            ybins = np.linspace(yamlConfig['MinY'], yamlConfig['MaxY'], yamlConfig['BinsY'] + 1)

            x = features_df_i[features[0]]
            y = features_df_i[features[1]]
            w = features_df_i[features[2]]

            hist, xedges, yedges = np.histogram2d(x, y, weights=w, bins=(xbins, ybins))

            for ix in range(0, yamlConfig['BinsX']):
                for iy in range(0, yamlConfig['BinsY']):
                    features_2dval[i, ix, iy, 0] = hist[ix, iy]
        features_val = features_2dval

    X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2,
                                                                random_state=42)

    # Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' and yamlConfig['InputType'] != 'Conv2D':
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)

    # Normalize inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] != 'Conv1D' and yamlConfig['InputType'] != 'Conv2D' and \
            yamlConfig['KerasLoss'] == 'squared_hinge':
        scaler = preprocessing.MinMaxScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
        y_train_val = y_train_val * 2 - 1
        y_test = y_test * 2 - 1

    # Normalize conv inputs
    if yamlConfig['NormalizeInputs'] and yamlConfig['InputType'] == 'Conv1D':
        reshape_X_train_val = X_train_val.reshape(X_train_val.shape[0] * X_train_val.shape[1], X_train_val.shape[2])
        scaler = preprocessing.StandardScaler().fit(reshape_X_train_val)
        for p in range(X_train_val.shape[1]):
            X_train_val[:, p, :] = scaler.transform(X_train_val[:, p, :])
            X_test[:, p, :] = scaler.transform(X_test[:, p, :])

    if 'j_index' in labels:
        labels = labels[:-1]

    return X_train_val, X_test, y_train_val, y_test, labels

## Config module
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.FullLoader)



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='', help='location of data to train off of')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_simple/', help='output directory')
    parser.add_option('-t','--test'   ,action='store',type='string',dest='test'   ,default='', help='Location of test data set')
    parser.add_option('-l','--load', action='store', type='string', dest='modelLoad', default=None, help='Model to load instead of training new')
    parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='configs/train_config_threelayer.yml', help='tree name')
    parser.add_option('-e','--epochs'   ,action='store',type='int', dest='epochs', default=100, help='number of epochs to train for')
    (options,args) = parser.parse_args()
    #print(options.config)
    yamlConfig = parse_config(options.config)

    current_model = models.three_layer_model()
    #current_model = models.three_layer_model_bv()
    #current_model = models.three_layer_model_seq(16,5)
   # summary(current_model,torch.zeros(16))
    #current_model.() #compains about getting doubles when expecting floats without this. Might be a problem with quantization, but dtypes *should* be handled better then



    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available() #False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    #X_train_val, X_test, y_train_val, y_test, labels = get_features(options,yamlConfig)
    L1_alpha = yamlConfig['L1RegR'] if 'L1RegR' in yamlConfig else 0.01  # Keras default value if not specified
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()
    L1_Loss = nn.L1Loss()
    optimizer = optim.Adam(current_model.parameters(), lr=0.0001, weight_decay=1e-5) #l2 weight reg since L1 is a bit more of a pain to implement
    batch_size = 200
    full_dataset = jet_dataset.ParticleJetDataset(options.inputFile,yamlConfig)
    test_dataset = jet_dataset.ParticleJetDataset(options.test, yamlConfig)
    train_size = int(0.75 * len(full_dataset)) #25% for Validation set, 75% for train set
    val_size = len(full_dataset) - train_size
    num_val_batches = math.ceil(val_size/batch_size)
    num_train_batches = math.ceil(train_size/batch_size)
    print("train_batches " + str(num_train_batches))
    print("val_batches " + str(num_val_batches))
    train_dataset, val_dataset =  torch.utils.data.random_split(full_dataset,[train_size,val_size])#Figure out data loading
    print("train dataset size: " + str(len(train_dataset)))
    print("validation dataset size: " + str(len(val_dataset)))
    print("test dataset size: " + str(len(test_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0) #FFS, have to use numworkers = 0 because apparently h5 objects can't be pickled, https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch/issues/69

    val_loader =   torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_size,
                                              shuffle=False, num_workers=0)
    val_losses = []
    train_losses = []
    roc_auc_scores = []
    avg_precision_scores = []
    accuracy_scores = []
    current_model.to(device)

    for epoch in range(options.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # Training
        #for local_batch, local_labels in train_loader:
        for i, data in enumerate(train_loader, 0):
            local_batch, local_labels = data
            current_model.train()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = current_model(local_batch.float())
            criterion_loss = criterion(outputs,local_labels.float())
            #criterion_loss = criterion(outputs, torch.max(local_labels, 1)[1]) #via https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/2
            #l1 = L1_Loss(outputs, torch.max(local_labels, 1)[1])
            #l1 reg on weights

            #L1_reg = torch.tensor(0., requires_grad=True)
            #for name, param in current_model.named_parameters():
            #    if 'weight' in name:
            #        L1_reg = L1_reg + torch.norm(param, 1)
            total_loss = criterion_loss# + (L1_reg*L1_alpha)
            total_loss.backward()
            optimizer.step()
            step_loss = total_loss.item()
            #train_losses.append(step_loss)
            if i == num_train_batches-1: #print every 8 batches for less console spam
                print('[epoch %d, batch: %1d] train batch loss: %.7f' % (epoch + 1, i + 1, step_loss))
                train_losses.append(step_loss)
            # Validation
        with torch.set_grad_enabled(False):
            current_model.eval()
            for i, data in enumerate(val_loader, 0):
                local_batch, local_labels = data
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                outputs = current_model(local_batch.float())
                #l1 = L1_Loss(outputs, torch.max(local_labels, 1)[1])
                #for name, param in current_model.named_parameters():
                #    if 'weight' in name:
                #        L1_reg = L1_reg + torch.norm(param, 1)
                val_loss = criterion(outputs,local_labels.float())#  + (L1_alpha * L1_reg)
                #val_loss = criterion(outputs, torch.max(local_labels, 1)[1])  # + (L1_alpha * L1_reg)
                #print(local_labels.numpy())
                #print(outputs.numpy())
                local_batch, local_labels = local_batch.cpu(), local_labels.cpu()
                outputs = outputs.cpu()
                val_roc_auc_score = roc_auc_score(local_labels.numpy(), outputs.numpy())
                val_avg_precision = average_precision_score(local_labels.numpy(), outputs.numpy())
                #roc_auc_scores.append(val_roc_auc_score)
                if i == num_val_batches-1:  # print every 8 batches for less console spam
                    print('[epoch %d, val batch: %1d] val batch loss: %.7f' % (epoch + 1, i + 1, val_loss))
                    print('[epoch %d, val batch: %1d] val ROC AUC Score: %.7f' % (epoch + 1, i + 1, val_roc_auc_score))
                    print('[epoch %d, val batch: %1d] val Avg Precision Score: %.7f' % (epoch + 1, i + 1, val_avg_precision))
                    val_losses.append(val_loss)
                    roc_auc_scores.append(val_roc_auc_score)
                    avg_precision_scores.append(val_avg_precision)
    now = datetime.now()
    time = now.strftime("%d-%m-%Y_%H-%M-%S")
    print("ROC AUC Table size: " + str(len(roc_auc_scores)))
    plt.plot(train_losses,color='r',linestyle='solid', alpha=0.3)
    plt.plot(val_losses, color='g',linestyle='dashed')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper left')
    plt.ylabel("Batch Loss (Per " + str(batch_size) + " Samples)")
    plt.xlabel("Epoch")
    plt.savefig(options.outputDir + 'loss_' + str(time) +'.png')
    plt.show()
    plt.plot(roc_auc_scores,color='r',linestyle='solid', alpha=0.3)
    plt.ylabel("ROC AUC")
    plt.xlabel("Epoch")
    plt.savefig(options.outputDir + 'ROCAUC_' + str(time) + '.png')
    plt.show()
    plt.plot(avg_precision_scores,color='r',linestyle='solid', alpha=0.3)
    plt.ylabel("Avg Precision")
    plt.xlabel("Epoch")
    plt.savefig(options.outputDir + 'avgPrec_' + str(time) + '.png')
    plt.show()

    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')
    outlist = torch.zeros(0, dtype=torch.double, device='cpu')
    prob_labels = torch.zeros(0, dtype=torch.double, device='cpu')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            current_model.eval()
            local_batch, local_labels = data
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = current_model(local_batch.float())
            _, preds = torch.max(outputs, 1)
            # Append batch prediction results
            outlist = torch.cat([outlist, outputs.cpu().type(torch.DoubleTensor)]).type(torch.DoubleTensor)
            prob_labels = torch.cat([prob_labels, local_labels.cpu().type(torch.DoubleTensor)]).type(torch.DoubleTensor)
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, torch.max(local_labels, 1)[1].view(-1).cpu()])
            #print(lbllist)
            #print(predlist)
            lo = prob_labels.cpu()
            outlist = outlist.cpu()
#        val_roc_curve = roc_curve(prob_labels.numpy(), outlist.numpy())

        outputs = outputs.cpu()
        local_labels = local_labels.cpu()
        predict_test = outputs.numpy()
        print(len(predict_test))
        df = pd.DataFrame()
        fpr = {}
        tpr = {}
        auc1 = {}

        plt.figure()
        for i, label in enumerate(full_dataset.labels_list):
            print(str(i) + " " + label)
            print(predict_test[:, i])
            df[label] = local_labels[:, i]
            print(df)
            print(len(predict_test[:, i]))
            df[label + '_pred'] = predict_test[:, i]

            fpr[label], tpr[label], threshold = roc_curve(df[label], df[label + '_pred'])

            auc1[label] = auc(fpr[label], tpr[label])

            plt.plot(tpr[label], fpr[label],
                     label='%s tagger, AUC = %.1f%%' % (label.replace('j_', ''), auc1[label] * 100.))
        plt.semilogy()
        plt.xlabel("Signal Efficiency")
        plt.ylabel("Background Efficiency")
        plt.ylim(0.001, 1)
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.figtext(0.25, 0.90, 'hls4ml', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.savefig(options.outputDir + 'ROC_' + str(time) + '.png' % ())

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    df_cm = pd.DataFrame(conf_mat, index=[i for i in full_dataset.labels_list],
                         columns=[i for i in full_dataset.labels_list])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True,fmt='g')
    plt.savefig(options.outputDir + 'confMatrix_' + str(time) + '.png')
    plt.show()
    print(conf_mat)
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print(class_accuracy)

    torch.save(current_model.state_dict(), options.outputDir + 'JetClassifyModel_' + str(time) + '.pt')



    print('Finished Training')
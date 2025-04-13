import os
import time
import pickle
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from torch_geometric.data import DataLoader
from nilearn.connectome import ConnectivityMeasure



def get_correlation_matrix(timeseries, msr):
    correlation_measure = ConnectivityMeasure(kind=msr)
    return correlation_measure.fit_transform([timeseries])[0]

def creat_dataloader():
    with open(config.data_path ,"rb") as f:
        data = pickle.load(f)

    feature_data = data["feature"]
    label_data = data["label"]

    dataset = []
    for i in range(len(feature_data)):
        features = feature_data[i]
        features = torch.tensor((get_correlation_matrix(features, 'correlation')))
        labels = torch.tensor(label_data[i])
        dataset.append((features, labels))

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return train_loader


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def show_train_hist(hist, show = True, save = True, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
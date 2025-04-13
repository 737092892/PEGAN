import pickle
import torch
from nilearn.connectome import ConnectivityMeasure
from torch_geometric.data import DataLoader
from config import *

def get_correlation_matrix(timeseries, msr='correlation'):
    correlation_measure = ConnectivityMeasure(kind=msr)
    return correlation_measure.fit_transform([timeseries])[0]


def load_dataset(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    dataset = []
    for features, labels in zip(data["feature"], data["label"]):
        corr_matrix = get_correlation_matrix(features)
        dataset.append((
            torch.tensor(corr_matrix).unsqueeze(0),  # Add channel dimension
            torch.tensor(labels)
        ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
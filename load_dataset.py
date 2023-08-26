import numpy as np
import pandas as pd
import torch
import torch.utils.data
import copy
import pickle

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_npy(name, path):

    if name == 'METR-LA':
        data = np.load(f'{path}/METR-LA/METR-LA.npy')
        data = data.reshape((-1,data.shape[-1]))
        graph_name = '_'.join([name, 'graph'])
        graph = np.load(f'{path}/METR-LA/METR-LA_graph.npy')

    elif name == 'Pems-Bay':
        df = pd.read_hdf('{path}/PEMS-BAY/PEMS-BAY.h5')
        data = np.expand_dims(df.values, axis=-1)
        data = data.squeeze(-1)
        with open('{path}/PEMS-BAY/adj_PEMS-BAY.pkl','rb') as f:
            _, _, graph = pickle.load(f, encoding='iso-8859-1')

    train_slice = slice(None, int(0.7*data.shape[0]))
    valid_slice = slice(int(0.7*data.shape[0]), int(0.85*data.shape[0]))
    test_slice = slice(int(0.85*data.shape[0]), None)
    
    scaler = StandardScaler(mean=data[train_slice].mean(),std=data[train_slice].std())
    
    pred_lens = [3]
    return data, graph, train_slice, valid_slice, test_slice, scaler, pred_lens

def generate_continuous_mask(T, n=5, l=0.1):
    res = np.full(T, True, dtype=bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    t = np.random.randint(T-l+1)
    res[t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.2):
    return torch.from_numpy(np.random.binomial(1, p, size=(T, B))).to(torch.bool)

# spatial&temporal augmentation
def ST_Aug(window, aug_size, station_no, neighbor):

    base = window[:,station_no]
    mask = generate_continuous_mask(aug_size)

    view = copy.deepcopy(base[:aug_size])
    view[~mask] = 0
    t_target = base[-aug_size:]
    
    s_base = window[:,neighbor]
    s_target = s_base[:aug_size]

    return np.array([view, t_target, s_target])

def extract_windows(data, graph, L0):

    windows = []

    for i in range(0, data.shape[0] - L0, 50):
        base_window = np.array(data[i:i + L0])
        for j in range(data.shape[1]):
            top_k = 2
            top_k_idx=graph[j,:].argsort()[::-1][0:top_k]
            neighbor = top_k_idx[-1]
            #neighbor = np.where(graph[j,:]==np.max(graph[j,:]))
            views = ST_Aug(base_window, 200, j, neighbor)
            windows.append(views)
    
    windows = np.array(windows)

    return windows

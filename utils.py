import numpy as np
import time
import STB.forecasting as forecasting
import torch
import joblib
from torch.utils.data import DataLoader

def generate_pred_samples(features, data, pred_len):
    
    hist_len = 12
    data = data.transpose(0,1)
    
    features = features.cpu()
    features = features.detach().numpy()
    
    data = data.cpu()
    data = data.detach().numpy()
    
    n = data.shape[1]

    labels = np.stack([data[:,i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:,hist_len:]
    
    features = features[:-pred_len, :]
    return features, \
            labels.reshape(-1, labels.shape[2])
            
def cal_metrics(pred, target):
    mask = target != 0
    return {
        'RMSE':np.sqrt(((pred[mask] - target[mask]) ** 2).mean()),
        'MAE': np.abs(pred[mask] - target[mask]).mean(),
        'MAPE': np.fabs((target[mask]-pred[mask])/target[mask]).mean()
    }
    
def masked_cal_metrics(pred, target):
    mask = target > 5 
    return {
        'RMSE':np.sqrt(((pred[mask] - target[mask]) ** 2).mean()),
        'MAE': np.abs(pred[mask] - target[mask]).mean(),
        'MAPE': np.fabs((target[mask]-pred[mask])/target[mask]).mean()
    }

def encode(data, encoder, projector, device, window_size, sliding_length, batch_size):
      
    data = data.flatten()
    window_list = []
    repr_list = []
    for i in range(0, data.shape[0]-window_size+1, sliding_length):

        window = data[i:i+window_size]

        window_list.append(window)
        
    dataloader = DataLoader(window_list, batch_size=batch_size, shuffle=False)
    
    for i, window in enumerate(dataloader):
        window = window.unsqueeze(2)
        projection = projector(window)
        projection = projection.transpose(1, 2)
    
        repr = encoder(projection)
        repr = repr[:,:,-1]
        repr = repr.reshape((repr.shape[0],-1)) 
        repr_list.append(repr)
    
    all_repr = torch.Tensor()
    all_repr = all_repr.double()
    all_repr = all_repr.to(device)

    for i in range(len(repr_list)):
        if all_repr == torch.Size([]):
            all_repr = repr_list[i]
        else:
            all_repr = torch.cat([all_repr,repr_list[i]],dim=0)

    return all_repr   
    
def generate_sample_list(data, encoder, projector, train_slice, valid_slice, test_slice, pred_len, padding, device):
    
    train_feature_list = []
    val_feature_list = []
    test_feature_list = []
        
    train_labels_list = []
    val_labels_list = []
    test_labels_list = []
    
    for i in range (data.shape[-1]):
            
        _data = data[:,i].unsqueeze(1)

        with torch.no_grad():
            
            all_repr = encode(_data, encoder, projector, device, window_size=12, sliding_length=1, batch_size=64)
                
            train_repr = all_repr[:int(0.7*data.shape[0])-12+1,:]
            valid_repr = all_repr[int(0.7*data.shape[0]):int(0.85*data.shape[0])-12+1,:]
            test_repr = all_repr[int(0.85*data.shape[0]):,:]
                
            train_data = _data[train_slice]
            valid_data = _data[valid_slice]
            test_data = _data[test_slice]
                
            train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len)
            valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
            test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
            
        train_feature_list.append(train_features)
        val_feature_list.append(valid_features)
        test_feature_list.append(test_features)
            
        train_labels_list.append(train_labels)
        val_labels_list.append(valid_labels)
        test_labels_list.append(test_labels)
    
    train_feature_list = np.array(train_feature_list) #node * T-P-H * H*feature: 320 
    val_feature_list = np.array(val_feature_list)
    test_feature_list = np.array(test_feature_list)

    train_feature_list = np.vstack(train_feature_list)
    val_feature_list = np.vstack(val_feature_list)
    test_feature_list = np.vstack(test_feature_list)

    train_labels_list = np.array(train_labels_list) #node * T-P-H * H*pred_length: 3
    val_labels_list = np.array(val_labels_list)
    test_labels_list = np.array(test_labels_list)
        
    train_labels_list = np.vstack(train_labels_list)
    val_labels_list = np.vstack(val_labels_list)
    test_labels_list = np.vstack(test_labels_list)
    
    return train_feature_list, train_labels_list, val_feature_list, val_labels_list, test_feature_list, test_labels_list

def generate_all_sample_list(data, encoder, projector, pred_len, padding, device):
    
    feature_list = []
        
    labels_list = []
    
    for i in range (data.shape[-1]):
            
        _data = data[:,i].unsqueeze(1)
            
        all_repr = encode(_data, encoder, projector, device, window_size=12, sliding_length=1, batch_size=64)
            
        features, labels =  generate_pred_samples(all_repr, _data, pred_len, sliding_length=1, drop=0)
            
        feature_list.append(features)
            
        labels_list.append(labels)
   
    feature_list = np.array(feature_list) #node * T-P-H * H*feature: 320 
    feature_list = np.vstack(feature_list)
  
    labels_list = np.array(labels_list) #node * T-P-H * H*pred_length: 3
    labels_list = np.vstack(labels_list)
    
    return feature_list, labels_list

def prediction(lr, path, name, test_feature_list, test_labels_list, scaler, pred_len):
    
    ours_result = {}
    lr_infer_time = {}
    out_log = {}
    
    t = time.time()
    test_pred_list = lr.predict(test_feature_list)
    lr_infer_time[pred_len] = time.time() - t

    test_pred_inv = scaler.inverse_transform(test_pred_list)
    test_labels_inv = scaler.inverse_transform(test_labels_list)
    
    np.save(f'{path}/Results/{name}/test_pred_list_{str(pred_len)}.npy', test_pred_list)
    np.save(f'{path}/Results/{name}/test_labels_list_{str(pred_len)}.npy', test_labels_list)
    np.save(f'{path}/Results/{name}/test_pred_inv_{str(pred_len)}.npy', test_pred_inv)
    np.save(f'{path}/Results/{name}/test_labels_inv_{str(pred_len)}.npy', test_labels_inv)
        
    out_log[pred_len] = {
        'norm': test_pred_list,
        'raw': test_pred_inv,
        'norm_gt': test_labels_list,
        'raw_gt': test_labels_inv
    }
    ours_result[pred_len] = {
        'norm': cal_metrics(test_pred_list, test_labels_list),
        'raw': masked_cal_metrics(test_pred_inv, test_labels_inv)
    }
        
    eval_res = {
        'ours': ours_result,
        'lr_infer_time': lr_infer_time
    }
    return out_log, eval_res
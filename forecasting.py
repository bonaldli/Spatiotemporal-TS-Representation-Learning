import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib
import time
import argparse
import os

from utils import *
from load_dataset import *

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n",type=str, default="METR-LA",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--lamda", '-ld', type=int,
                        default=0.5, help="The hyperparameter of loss function. Default is 0.5")
    parser.add_argument("--missing_rate", '-mr', type=int,
                        default=0, help="The missing rate of test data")
    parser.add_argument("--device", '-d', type=int,
                        default=0, help="GPU to use")
    return parser.parse_args()

def fit_ridge(train_features, train_y, valid_features, valid_y, MAX_SAMPLES=100000):
    # If the training set is too large, subsample MAX_SAMPLES examples
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features, train_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        train_features = split[0]
        train_y = split[2]
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features, valid_y,
            train_size=MAX_SAMPLES, random_state=0
        )
        valid_features = split[0]
        valid_y = split[2]
    
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)
        score = np.sqrt(((valid_pred - valid_y) ** 2).mean()) + np.abs(valid_pred - valid_y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]
    
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)
    return lr

def eval_forecasting(path, name, device, lamda, encoder, projector, data, train_slice, valid_slice, test_slice, scaler, pred_lens):
    padding = 200
    
    t = time.time()
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    data = scaler.transform(data)
    data = torch.from_numpy(data)
    data = data.to(device)
    encoder = encoder.to(device)
    projector = projector.to(device)
    
    for pred_len in pred_lens:
        
        train_feature_list, train_labels_list, val_feature_list, val_labels_list, test_feature_list, test_labels_list = generate_sample_list(data, encoder, projector, train_slice, valid_slice, test_slice, pred_len, padding, device)

        t = time.time()
        lr = fit_ridge(train_feature_list, train_labels_list, val_feature_list, val_labels_list)
        lr_train_time[pred_len] = time.time() - t
        
        joblib.dump(lr, f'{path}/lr_model/{name}_{lamda}_{pred_len}_lr.model')
        lr = joblib.load(f'{path}/lr_model/{name}_{lamda}_{pred_len}_lr.model')

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
            'raw': cal_metrics(test_pred_inv, test_labels_inv)
        }
        
    eval_res = {
        'ours': ours_result,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    return out_log, eval_res

def main():

    args = parse_args()
    path = os.path.dirname(__file__)
    org_data, graph, train_slice, valid_slice, test_slice, scaler, pred_lens = load_npy(args.name, path+'/datasets')
    data = copy.deepcopy(org_data)
    device = args.device
    
    f =  open(f'{path}/model/STB({args.name})_{args.lamda}.model','rb')
    s = f.read()
    encoder = pickle.loads(s)
    encoder.to(device)
    
    f1 = open(f'{path}/model/STB_projector({args.name})_{args.lamda}.model','rb')
    s1 = f1.read()
    projector = pickle.loads(s1)
    projector.to(device)

    mask = generate_binomial_mask(B = data.shape[1], T=data[test_slice].shape[0], p=0)
    padding_mask = generate_binomial_mask(B = data.shape[1], T=data.shape[0] - data[test_slice].shape[0], p=args.missing_rate)
    mask = np.concatenate([padding_mask,mask], axis=0)
    data[mask] = 0
    
    out_log, eval_res = eval_forecasting(path, args.name, device, args.lamda, encoder, projector, data, train_slice, valid_slice, test_slice, scaler, pred_lens)
    print(eval_res)


if __name__ == "__main__":
    main()
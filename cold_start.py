import os.path as osp
import os

import argparse

import numpy as np

import torch
import joblib

from STB.utils import *
import pickle
from load_dataset import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n",type=str, default="MTR",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--lamda", '-ld', type=int,
                        default=0.5, help="The hyperparameter of loss function. Default is 0.5")
    parser.add_argument("--device", '-d', type=int,
                        default=0, help="GPU to use")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    path = os.path.dirname(__file__)
    _, _, _, _, _, scaler, pred_lens = load_npy(args.name, path+'/datasets')

    device = args.device
    
    f =  open(f'{path}/model/STB({args.name})_{args.lamda}.model','rb')
    s = f.read()
    encoder = pickle.loads(s)
    encoder.to(device)
    
    f1 = open(f'{path}/model/STB_projector({args.name})_{args.lamda}.model','rb')
    s1 = f1.read()
    projector = pickle.loads(s1)
    projector.to(device)
    
    new_data = np.load(f'/mnt/users/lwangda/mtr/datasets/inflow_Tensor_June.npy')
    new_data = new_data.reshape((-1,new_data.shape[-1]))
    new_station = new_data[:,[80,81,84]]
    print(new_station.shape)

    new_data = scaler.transform(new_data)
    new_data = torch.from_numpy(new_data)
    new_data = new_data.to(device)

    new_station = scaler.transform(new_station)
    new_station = torch.from_numpy(new_station)
    new_station = new_station.to(device)
    padding = 200
    
    for pred_len in pred_lens:
        
        feature_list, labels_list = generate_all_sample_list(new_station, encoder, projector, pred_len, padding, device)
        lr = joblib.load(f'{path}/lr_model/{args.name}_{args.lamda}_{pred_len}_lr.model')
        out, eval_res = prediction(lr, feature_list, labels_list, scaler, pred_len)
        print(eval_res)
import numpy as np

import torch
from torch import optim
from tensorboardX import SummaryWriter
torch.manual_seed(0)

from STB import STB, LogisticRegression

from load_dataset import *
from torch.utils.data import DataLoader
import argparse
import os

import pickle
#import main 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data",
                        help="Path to data directory, where all the datasets will be placed. Default is 'data'")
    parser.add_argument("--name", "-n",type=str, default="METR-LA",
                        help="Name of the dataset. Supported names are: cora, citeseer, pubmed, photo, computers, cs, and physics")
    parser.add_argument("--layers", "-l", nargs="+", default=[
                        512, 256], help="The number of units of each layer of the GNN. Default is [512, 128]")
    parser.add_argument("--hidden_dims", '-hd', type=int,
                        default=64, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--pred_hid", '-ph', type=int,
                        default=512, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--lr", '-lr', type=float, default=0.00001,
                        help="Learning rate. Default is 0.0001.")
    parser.add_argument("--dropout", "-do", type=float,
                        default=0.0, help="Dropout rate. Default is 0.2")
    parser.add_argument("--cache-step", '-cs', type=int, default=300,
                        help="The step size to cache the model, that is, every cache_step the model is persisted. Default is 100.")
    parser.add_argument("--epochs", '-e', type=int,
                        default=20, help="The number of epochs")
    parser.add_argument("--lamda", '-ld', type=int,
                        default=0.5, help="The hyperparameter of loss function. Default is 0.5")
    parser.add_argument("--device", '-d', type=int,
                        default=0, help="GPU to use")
    return parser.parse_args()


class ModelTrainer:

    def __init__(self, args):
        self._args = args
        self._init()
        self.writer = SummaryWriter(log_dir=os.path.dirname(__file__)+"/runs/STB_dataset({})".format(args.name))

    def _init(self):
        args = self._args
        self._device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")

        self._model = STB(input_dims=1, output_dims=320, lamda=args.lamda, hidden_dims=args.hidden_dims, pred_hid=args.pred_hid, dropout=args.dropout, epochs=args.epochs).to(self._device)
        self._model = self._model.double()

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay= 1e-5)
        # learning rate
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
                    else ( 1 + np.cos((epoch-1000) * np.pi / (self._args.epochs - 1000))) * 0.5
        self._scheduler = optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda = scheduler)

    def train(self, train_data, path):
        # get initial test results
        print("start training!")
        print("Initial Evaluation...")

        #data augmentation
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, drop_last=True)

        # start training        
        self._model.train()
        for epoch in range(self._args.epochs):

            for batch in train_loader:
                
                view = batch[:,0,:].unsqueeze(2) #B T Ch
                t_target = batch[:,1,:].unsqueeze(2)
                s_target = batch[:,2,:].unsqueeze(2)

                view = view.to(self._device)
                t_target = t_target.to(self._device)
                s_target = s_target.to(self._device)

                self._model = self._model.to(self._device)

                output, loss = self._model(view=view, t_target=t_target, s_target=s_target)
                    
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._scheduler.step()
                self._model.update_moving_average()

            #sys.stdout.write('\rEpoch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, loss.data, self._optimizer.param_groups[0]['lr']))
            print('\rEpoch {}/{}, loss {:.4f}, lr {}'.format(epoch + 1, self._args.epochs, loss.data, self._optimizer.param_groups[0]['lr']))
        
      
        s = pickle.dumps(self._model.student_encoder)
        with open(f'{path}/model/STB({self._args.name})_{self._args.lamda}.model','wb+') as f:
            f.write(s)
        s = pickle.dumps(self._model.input_fc)
        with open(f'{path}/model/STB_projector({self._args.name})_{self._args.lamda}.model','wb+') as f:
            f.write(s)

        print("Training Done!")
        
def train_eval(args):

    path = os.path.dirname(__file__)
    data, graph, train_slice, _, _, _, _ = load_npy(args.name, path+'/datasets')
    train_data = extract_windows(data[train_slice,:],graph, L0=250) 
    trainer = ModelTrainer(args)
    trainer.train(train_data,path)    
    trainer.writer.close()


def main():
    args = parse_args()
    train_eval(args)


if __name__ == "__main__":
    main()
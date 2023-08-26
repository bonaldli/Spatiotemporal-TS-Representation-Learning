import torch
from torch import nn
import torch.nn.functional as F

from network import DilatedConvEncoder

import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class STB(nn.Module):

    def __init__(self, input_dims, output_dims, lamda=0.5, encode_length=200, hidden_dims=64, depth=10, pred_hid=64, dropout=0.0, moving_average_decay=0.99, epochs=1000):
        super().__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.student_encoder = DilatedConvEncoder(hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        self.teacher_encoder = DilatedConvEncoder(hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        self.student_predictor = nn.Sequential(nn.Linear(output_dims*encode_length, pred_hid), nn.PReLU(), nn.Linear(pred_hid, output_dims*encode_length))
        self.student_predictor.apply(init_weights)
        self.repr_dropout = nn.Dropout(p=dropout)
        self.lamda = lamda
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, view, t_target, s_target):
        hid_student = self.input_fc(view)
        hid_student = hid_student.transpose(1, 2) #B Ch T
        student = self.student_encoder(x=hid_student)

        student = student.view((student.shape[0],-1))
        pred = self.student_predictor(student)
        
        with torch.no_grad():

            hid_t_teacher = self.input_fc(t_target)
            hid_s_teacher = self.input_fc(s_target)

            hid_t_teacher = hid_t_teacher.transpose(1, 2)
            hid_s_teacher = hid_s_teacher.transpose(1, 2)

            t_teacher = self.teacher_encoder(x=hid_t_teacher)
            s_teacher = self.teacher_encoder(x=hid_s_teacher)

            t_teacher = t_teacher.reshape((t_teacher.shape[0],-1))
            s_teacher = s_teacher.reshape((s_teacher.shape[0],-1))
            
        loss1 = loss_fn(pred, t_teacher.detach())
        loss2 = loss_fn(pred, s_teacher.detach())

        loss = self.lamda*loss1 + (1-self.lamda)*loss2
        return student, loss.mean()


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):

        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)

        return logits, loss

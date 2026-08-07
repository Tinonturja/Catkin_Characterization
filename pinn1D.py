import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import data_df
class PINN1D(nn.Module):
    def __init__(self,echos:int, loss_fn:torch.nn.Module = torch.nn.MSELoss(),
                 optimizer:torch.optim.Optimizer = torch.optim.Adam, 
                 learning_rate:float = 0.001):
        super(PINN1D, self).__init__()
        self.echos = echos
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.LayerNorm(20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.LayerNorm(20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        # call the learnable parameters of the network
        self.qe_ = nn.Parameter(torch.tensor(0.0, requires_grad=True)) # later apply softmax on this parameter to ensure it is positive
        self.k2_ = nn.Parameter(torch.tensor(0.0, requires_grad=True))

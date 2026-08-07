import torch
import matplotlib.pyplot as plt

def set_seed(seed = 0):
    torch.manual_seed(seed)

def plot_loss(history):
    
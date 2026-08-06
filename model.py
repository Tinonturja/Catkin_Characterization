import torch
import torch.nn as nn
import torch.nn.functional as F

class AdsorptionModelInversePINN(nn.Module):
    def __init__(self,hidden_layer:int,
                 input_layer:int,
                 output_layer:int):
        super().__init__()
        self.learned_qe_ = nn.Parameter(requires_grad = True)
        self.learned_k2_ = nn.Parameter(requires_grad = True) 
        self.net = nn.Sequential(
            nn.Linear(input_layer,
                      hidden_layer),
            nn.LayerNorm(hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer,
                      hidden_layer),
            nn.LayerNorm(hidden_layer),
            nn.Tanh(),
            nn.Linear(hidden_layer,
                      output_layer)

        )



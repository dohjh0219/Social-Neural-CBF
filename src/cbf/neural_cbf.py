import torch
import torch.nn as nn

class NeuralCBFNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(NeuralCBFNetwork, self).__init__()
        
        # input : 상대위치 x, y, 상대속도 vx, vy
        # output : 1개 (safety score h)
        
        # MLP(3 Layers)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Layer 1
            nn.ReLU(),                        # Activation (Nonlinearity)
            nn.Linear(hidden_dim, hidden_dim),# Layer 2 : hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)          # Output Layer (h값 하나 출력)
        )

    def forward(self, x):
        return self.net(x)
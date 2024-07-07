import torch.nn as nn
from config.metropolis_helper import *
from config.pickle_helper import *


# Define the same PBNN model structure as used during training
class DoublePendulumPBNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DoublePendulumPBNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
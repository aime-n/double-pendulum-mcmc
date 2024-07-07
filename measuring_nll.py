from config.metropolis_helper import *
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from double_pendulum import DoublePendulumDataset
from tqdm import tqdm
import os
from config.pickle_helper import *


# Double pendulum dataset initialization
max_time = 20
dt = 0.02
initial_condition = (np.pi / 2, 5.0, np.pi, 5.0)
batch_size = 64
number_of_epochs = 1000
weight_decay = 10.0

dataset = DoublePendulumDataset(max_time=max_time, dt=dt, mass_1=1.0, mass_2=1.0, initial_condition=initial_condition)
train_loader, val_loader = dataset.get_train_test_data_loaders(batch_size=batch_size, validation_split=0.7)

# Load the samples
samples = load_samples('samples.pkl')

avg_nll = compute_nll_mini_batch(val_loader, samples, batch_size)
print(f'Average Negative Log-Likelihood: {avg_nll}')

save_nll(avg_nll)
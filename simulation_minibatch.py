import numpy as np
from config.metropolis_helper import *
import torch
import matplotlib.pyplot as plt
from double_pendulum import DoublePendulumDataset
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

initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
samples = metropolis_hastings(initial_state, train_loader, iterations=10000, save_interval=1000, save_path='samples.pkl')

samples = torch.stack(samples)
burn_in = 1000
samples = samples[burn_in:]

samples_np = samples.numpy()
if samples_np.size == 0:
    print("No samples generated.")
else:
    plt.plot(samples_np[:, 0], label='Theta1')
    plt.plot(samples_np[:, 1], label='Theta2')
    plt.legend()

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, 'mcmc_samples.png'))
    plt.close()

    print(f"Plot saved to {os.path.join(output_dir, 'mcmc_samples.png')}")
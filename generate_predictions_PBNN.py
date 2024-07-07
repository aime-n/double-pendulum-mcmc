import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from double_pendulum import DoublePendulumDataset
from config.metropolis_helper import *
from config.pickle_helper import *
from config.model import DoublePendulumPBNN


# Loss function incorporating physical constraints (regularization term)
def physics_loss(output, target, state, weight_decay):
    prediction_loss = nn.MSELoss()(output, target)
    # Add a regularization term based on physical constraints if necessary
    # For example, this could be based on the known equations of motion
    regularization = weight_decay * torch.sum(torch.square(output - state))  # Example
    return prediction_loss + regularization

# Load the dataset
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
samples = torch.stack(samples)
burn_in = 1000
samples = samples[burn_in:]

# Convert samples to dataset
train_data = TensorDataset(samples[:-1], samples[1:])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
input_size = 4  # State of the double pendulum (theta1, theta2, omega1, omega2)
hidden_size = 64
output_size = 4  # Predicted next state
model = DoublePendulumPBNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(number_of_epochs):
    model.train()
    epoch_loss = 0
    for state, next_state in train_loader:
        state = state.float()
        next_state = next_state.float()
        
        optimizer.zero_grad()
        output = model(state)
        loss = physics_loss(output, next_state, state, weight_decay)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{number_of_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), 'double_pendulum_pbnn_model.pth')
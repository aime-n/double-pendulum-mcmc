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

def calculate_accuracy(y_true, y_pred, threshold=0.10):
    """
    Calculate the percentage of predictions within a certain percentage threshold of the true values.
    
    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        threshold (float): Percentage error threshold for correctness.
        
    Returns:
        float: Percentage of correct predictions.
    """
    # Calculate the absolute percentage error
    percentage_error = np.abs((y_pred - y_true) / y_true)
    
    # Determine which predictions are within the threshold
    correct_predictions = percentage_error <= threshold
    
    # Calculate the accuracy
    accuracy = np.mean(correct_predictions) * 100
    
    return accuracy

# Load the dataset
max_time = 20
dt = 0.02
initial_condition = (np.pi / 2, 5.0, np.pi, 5.0)
batch_size = 64
validation_split = 0.7

dataset = DoublePendulumDataset(max_time=max_time, dt=dt, mass_1=1.0, mass_2=1.0, initial_condition=initial_condition)
_, val_loader = dataset.get_train_test_data_loaders(batch_size=batch_size, validation_split=validation_split)

# Initialize the model
input_size = 4  # State of the double pendulum (theta1, theta2, omega1, omega2)
hidden_size = 64
output_size = 4  # Predicted next state
model = DoublePendulumPBNN(input_size, hidden_size, output_size)

# Load the trained model weights
model.load_state_dict(torch.load('double_pendulum_pbnn_model.pth'))
model.eval()

# Prepare data for plotting and accuracy calculation
true_states = []
predicted_states = []

with torch.no_grad():
    for state, next_state in val_loader:
        state = state.float().view(-1, input_size)  # Ensure correct input dimensions
        next_state = next_state.float().view(-1, output_size)  # Ensure correct output dimensions
        output = model(state)
        true_states.append(next_state)
        predicted_states.append(output)

true_states = torch.cat(true_states)
predicted_states = torch.cat(predicted_states)

# Ensure true_states and predicted_states have the same length
min_length = min(true_states.size(0), predicted_states.size(0))
true_states = true_states[:min_length].numpy()
predicted_states = predicted_states[:min_length].numpy()

# Correcting the length of time_steps array
time_steps = np.arange(0, true_states.shape[0] * dt, dt)

# Ensure time_steps has the same length as true_states and predicted_states
time_steps = time_steps[:true_states.shape[0]]

fig, axs = plt.subplots(4, 1, figsize=(10, 15))

for i, label in enumerate(['theta1', 'theta2', 'omega1', 'omega2']):
    axs[i].plot(time_steps, true_states[:, i], label=f'True {label}')
    axs[i].plot(time_steps, predicted_states[:len(time_steps), i], label=f'Predicted {label}', linestyle='--')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel(label)
    axs[i].legend()

plt.tight_layout()
plt.savefig('double_pendulum_predictions.png')
plt.show()

# Calculate and print accuracy
accuracy = calculate_accuracy(true_states, predicted_states)
print(f"Prediction Accuracy: {accuracy:.2f}%")

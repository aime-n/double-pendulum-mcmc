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

class DoublePendulumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DoublePendulumLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # Hidden state
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)  # Cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Custom loss function with physics-based constraints
def physics_loss(output, target, state, weight_decay):
    prediction_loss = nn.MSELoss()(output, target)
    # Regularization term (example: ensuring small changes in state)
    regularization = weight_decay * torch.sum((output - state) ** 2)
    return prediction_loss + regularization

# Load the dataset
max_time = 20
dt = 0.02
initial_condition = (np.pi / 2, 5.0, np.pi, 5.0)
batch_size = 64
validation_split = 0.7

dataset = DoublePendulumDataset(max_time=max_time, dt=dt, mass_1=1.0, mass_2=1.0, initial_condition=initial_condition)
train_loader, val_loader = dataset.get_train_test_data_loaders(batch_size=batch_size, validation_split=validation_split)

# Initialize the model
input_size = 4  # State of the double pendulum (theta1, theta2, omega1, omega2)
hidden_size = 64
output_size = 4  # Predicted next state
num_layers = 2
model = DoublePendulumLSTM(input_size, hidden_size, output_size, num_layers)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load the trained model weights if available
model_path = 'double_pendulum_lstm_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(model.parameters(), lr=0.001)
weight_decay = 10.0
number_of_epochs = 1000

# Training loop
for epoch in range(number_of_epochs):
    model.train()
    epoch_loss = 0
    for state, next_state in train_loader:
        state, next_state = state.to(device), next_state.to(device)
        state = state.view(-1, 1, input_size)
        next_state = next_state.view(-1, output_size)

        optimizer.zero_grad()
        output = model(state)
        loss = physics_loss(output, next_state, state.view(-1, output_size), weight_decay)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{number_of_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

# Save the model
torch.save(model.state_dict(), model_path)

# Validation
model.eval()
true_states = []
predicted_states = []

with torch.no_grad():
    for state, next_state in val_loader:
        state, next_state = state.to(device), next_state.to(device)
        state = state.view(-1, 1, input_size)
        next_state = next_state.view(-1, output_size)
        
        output = model(state)
        true_states.append(next_state.cpu())
        predicted_states.append(output.cpu())

true_states = torch.cat(true_states).numpy()
predicted_states = torch.cat(predicted_states).numpy()

# Correcting the length of time_steps array
time_steps = np.arange(0, true_states.shape[0] * dt, dt)
time_steps = time_steps[:true_states.shape[0]]

fig, axs = plt.subplots(4, 1, figsize=(10, 15))

for i, label in enumerate(['theta1', 'theta2', 'omega1', 'omega2']):
    axs[i].plot(time_steps, true_states[:, i], label=f'True {label}')
    axs[i].plot(time_steps[:len(true_states)], predicted_states[:len(true_states), i], label=f'Predicted {label}', linestyle='--')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel(label)
    axs[i].legend()

plt.tight_layout()
plt.savefig('double_pendulum_predictions_lstm.png')
plt.show()

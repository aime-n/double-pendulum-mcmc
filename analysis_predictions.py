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

# avg_nll = compute_nll_mini_batch(val_loader, samples, batch_size)
avg_nll = load_pickles(filename='nll.pkl')
print(f'Average Negative Log-Likelihood: {avg_nll}')

# Generate predictions for the test data
# predictions = generate_predicions(val_loader=val_loader, samples=samples)
predictions = load_pickles(filename='predictions.pkl')

# Ensure predictions have the correct shape
predictions = np.array(predictions)
print(f"Predictions shape: {predictions.shape}")

if len(predictions.shape) == 3:
    mean_predictions = predictions.mean(axis=2)
    std_predictions = predictions.std(axis=2)
elif len(predictions.shape) == 2:
    # Reshape predictions to 3D if they are 2D
    predictions = np.expand_dims(predictions, axis=2)
    mean_predictions = predictions.mean(axis=2)
    std_predictions = predictions.std(axis=2)
else:
    # If predictions do not have 2 or 3 dimensions, raise an error or handle accordingly
    raise ValueError("Predictions array does not have 2 or 3 dimensions.")

time_steps = np.arange(len(mean_predictions))

plt.figure(figsize=(14, 5))
plt.plot(time_steps, val_loader.dataset.tensors[0][:, 0], label='True Theta1', color='blue')
plt.plot(time_steps, mean_predictions[:, 0], label='Mean Prediction Theta1', color='red')
plt.fill_between(time_steps, 
                 mean_predictions[:, 0] - std_predictions[:, 0], 
                 mean_predictions[:, 0] + std_predictions[:, 0], 
                 color='gray', alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Theta1')
plt.legend()

output_dir = 'output'

plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
plt.close()

print(f"Prediction plot saved to {os.path.join(output_dir, 'test_predictions.png')}")

# Save all computations in a pickle file
results = {
    'samples': samples,
    # 'avg_nll': avg_nll,
    'predictions': predictions,
    'mean_predictions': mean_predictions,
    'std_predictions': std_predictions,
}

save_results(results, 'results.pkl')

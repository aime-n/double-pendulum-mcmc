import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from double_pendulum import DoublePendulumDataset
from tqdm import tqdm

# Define the log-likelihood function
def log_likelihood(state, data):
    theta1, theta2, omega1, omega2 = state
    log_likelihood = 0
    for observation in data:
        theta1_obs, theta2_obs, omega1_obs, omega2_obs = observation[:4]
        log_likelihood += -0.5 * ((theta1 - theta1_obs)**2 + (theta2 - theta2_obs)**2 +
                                  (omega1 - omega1_obs)**2 + (omega2 - omega2_obs)**2)
    return log_likelihood

# Proposal distribution: Gaussian centered at current state
def proposal(state, step_size=0.1):
    return state + torch.normal(0, step_size, size=state.shape)

# Metropolis-Hastings algorithm with mini-batch sampling with replacement
def metropolis_hastings(initial_state, data_loader, iterations, batch_size=60, step_size=0.1):
    current_state = initial_state
    samples = [current_state]
    
    for _ in tqdm(range(iterations), desc="Metropolis-Hastings Sampling"):
        mini_batch = next(iter(data_loader))[0]
        
        proposed_state = proposal(current_state, step_size)
        log_acceptance_ratio = (log_likelihood(proposed_state, mini_batch) - 
                                log_likelihood(current_state, mini_batch))
        
        if np.log(np.random.rand()) < log_acceptance_ratio:
            current_state = proposed_state
            
        samples.append(current_state)
    
    return torch.stack(samples)

# Function to compute the negative log-likelihood for a mini-batch
def compute_nll_mini_batch(data_loader, model_params_samples, batch_size):
    n_batches = len(data_loader.dataset) // batch_size
    nll_total = 0
    L = len(data_loader.dataset)
    J = len(model_params_samples)
    
    for batch_index, (batch_data, _) in enumerate(data_loader):
        if batch_index >= n_batches:
            break
        
        for observation in batch_data:
            log_prob_sum = 0
            for j in range(J):
                theta_j = model_params_samples[j]
                p_y_given_x_theta = model_prediction_prob(observation, theta_j)
                log_prob_sum += p_y_given_x_theta
            
            log_prob_avg = log_prob_sum / J
            nll_total -= torch.log(log_prob_avg)
    
    avg_nll = nll_total / L
    return avg_nll.item()

# Define a placeholder model prediction function
def model_prediction_prob(observation, params):
    theta1, theta2, omega1, omega2 = params
    theta1_obs, theta2_obs, omega1_obs, omega2_obs = observation[:4]
    prob = torch.exp(-0.5 * ((theta1 - theta1_obs)**2 + (theta2 - theta2_obs)**2 +
                             (omega1 - omega1_obs)**2 + (omega2 - omega2_obs)**2))
    return prob

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
samples = metropolis_hastings(initial_state, train_loader, iterations=10000)
burn_in = 1000
samples = samples[burn_in:]

print(f"Generated {len(samples)} samples after burn-in")

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

avg_nll = compute_nll_mini_batch(val_loader, samples, batch_size)
print(f'Average Negative Log-Likelihood: {avg_nll}')

# Generate predictions for the test data
test_loader = val_loader  # Assuming val_loader is used as test data here
predictions = []

for batch_data, _ in test_loader:
    batch_predictions = []
    for observation in batch_data:
        observation_predictions = [model_prediction_prob(observation, params).item() for params in samples]
        batch_predictions.append(observation_predictions)
    predictions.append(batch_predictions)

predictions = np.array(predictions)
print(f"Predictions shape: {predictions.shape}")

mean_predictions = predictions.mean(axis=1)
std_predictions = predictions.std(axis=1)

time_steps = np.arange(len(mean_predictions))

plt.figure(figsize=(14, 5))
plt.plot(time_steps, test_loader.dataset.tensors[0][:, 0], label='True Theta1', color='blue')
plt.plot(time_steps, mean_predictions[:, 0], label='Mean Prediction Theta1', color='red')
plt.fill_between(time_steps, 
                 mean_predictions[:, 0] - std_predictions[:, 0], 
                 mean_predictions[:, 0] + std_predictions[:, 0], 
                 color='gray', alpha=0.5)
plt.xlabel('Time Steps')
plt.ylabel('Theta1')
plt.legend()

plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
plt.close()

print(f"Prediction plot saved to {os.path.join(output_dir, 'test_predictions.png')}")

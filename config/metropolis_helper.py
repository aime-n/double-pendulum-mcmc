import os
import numpy as np
import torch
from tqdm import tqdm
from config.pickle_helper import *


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
def metropolis_hastings(initial_state, data_loader, iterations, batch_size=60, step_size=0.1, save_interval=1000, save_path='samples.pkl'):
    current_state = initial_state
    samples = [current_state]
    
    for i in tqdm(range(iterations), desc="Metropolis-Hastings Sampling"):
        mini_batch = next(iter(data_loader))[0]
        
        proposed_state = proposal(current_state, step_size)
        log_acceptance_ratio = (log_likelihood(proposed_state, mini_batch) - 
                                log_likelihood(current_state, mini_batch))
        
        if np.log(np.random.rand()) < log_acceptance_ratio:
            current_state = proposed_state
            
        samples.append(current_state)
        
        if (i + 1) % save_interval == 0:
            save_samples(samples, save_path)
    
    save_samples(samples, save_path)
    return torch.stack(samples)

# Function to compute the negative log-likelihood for a mini-batch with progress bars
def compute_nll_mini_batch(data_loader, model_params_samples, batch_size):
    n_batches = len(data_loader.dataset) // batch_size
    nll_total = 0
    L = len(data_loader.dataset)
    J = len(model_params_samples)
    
    for batch_index, (batch_data, _) in enumerate(tqdm(data_loader, desc="NLL Computation (Batches)")):
        if batch_index >= n_batches:
            break
        
        for observation in tqdm(batch_data, desc="Observations", leave=False):
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

def generate_predicions(val_loader, samples):
    predictions = []

    for batch_data, _ in tqdm(val_loader, desc="Generating Predictions"):  # Assuming val_loader is used as test data here
        batch_predictions = []
        for observation in tqdm(batch_data, desc="Processing Observations", leave=False):
            observation_predictions = [model_prediction_prob(observation, params).item() for params in samples]
            batch_predictions.append(observation_predictions)
        predictions.append(batch_predictions)
    return predictions

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from double_pendulum import DoublePendulumDataset
import matplotlib.pyplot as plt

# Step 1: Load the dataset
dataset = DoublePendulumDataset()
train_loader, test_loader = dataset.get_train_test_data_loaders(batch_size=100)

# Step 2: Define the BNN
class BNN(nn.Module):
    def __init__(self):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Step 3: Define the loss function with noise penalty
def loss_fn(output, target, model, batch_size):
    nll_loss = nn.MSELoss()(output, target)
    prior_loss = 0.5 * sum(param.pow(2).sum() for param in model.parameters())
    noise_penalty = np.var(output.detach().cpu().numpy()) / batch_size
    return nll_loss + prior_loss + noise_penalty

# Step 4: Implement the MCMC Algorithm with Noise Penalty
def mcmc_sample(model, train_loader, num_samples=1000, burn_in=200):
    samples = []
    current_params = [param.clone().detach() for param in model.parameters()]
    current_loss = 0
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_samples + burn_in):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target, model, batch_size=len(data))
            loss.backward()
            optimizer.step()

            new_params = [param.clone().detach() for param in model.parameters()]
            new_loss = loss.item()

            acceptance_prob = min(1, np.exp(current_loss - new_loss))

            if np.random.rand() < acceptance_prob:
                current_params = new_params
                current_loss = new_loss

        if epoch >= burn_in:
            samples.append(current_params)

    return samples

# Step 5: Evaluate the Model
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target, model, batch_size=len(data)).item()
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')

# Main function to run the experiment
if __name__ == "__main__":
    model = BNN()
    samples = mcmc_sample(model, train_loader)

    # Evaluate using the last sample of the parameters
    for param, sample in zip(model.parameters(), samples[-1]):
        param.data.copy_(sample)

    evaluate(model, test_loader)

    # Optional: Save the samples for further analysis
    torch.save(samples, 'mcmc_samples.pth')

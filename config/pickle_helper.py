import pickle

# Save samples using pickle
def save_samples(samples, filename):
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)

# Load samples using pickle
def load_samples(filename):
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    return samples


def load_pickles(filename):
    with open(filename, 'rb') as f:
        plk = pickle.load(f)
    return plk

# Save all computations using pickle
def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

# Save predictions
def save_predictions(predictions, filename='predictions.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(predictions, f)

# Save NLL
def save_nll(nll, filename='nll.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(nll, f)
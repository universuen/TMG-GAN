import pickle

with open('data.pkl', 'rb') as f:
    training_samples, training_labels, test_samples, test_labels = pickle.load(f)

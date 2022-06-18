import pickle

with open('data_np.pkl', 'rb') as f:
    training_samples, training_labels, test_samples, test_labels = pickle.load(f)

print(set(test_labels))

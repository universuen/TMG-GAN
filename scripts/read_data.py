import pickle
from collections import Counter

with open('data.pkl', 'rb') as f:
    training_samples, training_labels, test_samples, test_labels = pickle.load(f)

print(set(test_labels))
print(Counter(training_labels))

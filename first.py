import numpy as np 
import csv

# Load MNIST data from CSV file
# Assuming the CSV file is formatted with each row as an image and the first column as the label
# For example, the first row might look like: [label, pixel1, pixel2    , ..., pixel784]                
with open("minist_train_100", "r") as f:
    data = []
    labels = []
    # Read the CSV file

    for line in f:
        values = line.split(",")
        label = int(values[0])               # first value is label
        pixels = [int(v) for v in values[1:]]  # rest is pixel data

        labels.append(label)
        data.append(pixels)

# Neural network weights
w1 = np.array([[0.5, 0.5, 0.25], [0.5, 0.25, 0.5], [0.5, 0.25, 0.5]])
w2 = np.array([[0.5, 0.25, 0.5], [0.5, 0.25, 0.5], [0.5, 0.25, 0.5]])

# Test input values
input_layer = np.array([1.0, 2.0, 3.0])
print("Input layer:", input_layer)

# Forward propagation
hidden_layer = np.dot(w1, input_layer)
print("Hidden layer (after w1):", hidden_layer)

output_layer = np.dot(w2, hidden_layer)
print("Output layer (after w2):", output_layer)

# Show all layers
print("\nAll layers:")
print("Input:", input_layer)
print("Hidden:", hidden_layer) 
print("Output:", output_layer)
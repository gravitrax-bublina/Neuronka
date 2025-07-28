import numpy as np 
import csv

# Load MNIST data from CSV file
# Assuming the CSV file is formatted with each row as an image and the first column as the label
# For example, the first row might look like: [label, pixel1, pixel2    , ..., pixel784]                
with open("Neuronka\mnist_train_100.csv", "r") as f:
    data = []
    labels = []
    # Read the CSV file

    for line in f:
        values = line.split(",")
        label = int(values[0])               # first value is label
        pixels = [int(v) for v in values[1:]]  # rest is pixel data

        labels.append(label)
        data.append(pixels)


def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


# Neural network weights
w1 = np.random.uniform(0, 0.05, (256, 784))  # 16 hidden neurons, 784 inputs
w2 = np.random.uniform(0, 0.1, (16, 256)) 
w3 = np.random.uniform(0, 0.5, (10, 16))    # 10 output neurons, 16 hidden neurons
# Test input values

# Forward propagation
hidden_layer1 = sigmoid(np.dot(w1, data[0]))  # Using the first label as input
hidden_layer2 = sigmoid(np.dot(w2, hidden_layer1))
output_layer = sigmoid(np.dot(w3, hidden_layer2))


#w3[0][0] = labels[0] -
print("Output layer (after w3):", output_layer)
er = []
nextgenlabels = []
for x in range(1):
    er.append([0] * len(output_layer))  # Initialize error list for each label
    for i in range(len(output_layer)):
        if i == labels[0]:
            nextgenlabels.append(1)
        else:
            nextgenlabels.append(0)
        
        if x == 0:
            er[x][i] = (nextgenlabels[i] - output_layer[i])**2
print(nextgenlabels)

print("Error for first label:", er[0])

chclanku = [hidden_layer1, hidden_layer2]
    
print(w3)
for x in range(len(w3)):
    for y in range(len(w3[x])):
        chclanku[x][y] = er[y]* (w3[x][y]/np.sum(w3[x]))

for xxxx in range(len(w3)):
    
    chclanku[xxxx] = np.sum(chclanku[xxxx])


        

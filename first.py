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
for asd in range(len(data)):
        
    # Forward propagation
    hidden_layer1 = sigmoid(np.dot(w1, data[asd]))  # Using the first label as input
    hidden_layer2 = sigmoid(np.dot(w2, hidden_layer1))
    output_layer = sigmoid(np.dot(w3, hidden_layer2))


    #w3[0][0] = labels[0] -
    print("Output layer (after w3):", output_layer)
    er = []
    nextgenlabels = []
    for x in range(1):
        er.append([0] * len(output_layer))  # dáváme desítky nul (jen jedna chic hic hichi)
        nextgenlabels = []
        
        for i in range(len(output_layer)):
            if i == labels[asd]:
                nextgenlabels.append(1)
            else:
                nextgenlabels.append(0)
            
            if x == 0:
                er[x][i] = (nextgenlabels[i] - output_layer[i])**2
    print(nextgenlabels)

    print("Error for first label:", er[0])

    chclanku = [
        np.zeros_like(w1),
        np.zeros_like(w2),
        np.zeros_like(w3)
    ]
    chneuronu = [hidden_layer1, hidden_layer2]

    for x in range(len(w3)):
        for y in range(len(w3[x])):
            chclanku[2][x][y] = er[0][x] * (w3[x][y]/np.sum(w3[x]))

    for xxxx in range(len(w3)):
        chneuronu[1][xxxx] = np.sum(chclanku[2][xxxx])


    for x in range(len(w2)):
        for y in range(len(w2[x])):
            chclanku[1][x][y] = chneuronu[1][x] * (w2[x][y]/np.sum(w2[x]))

    for xxxx in range(len(w2)):
        chneuronu[0][xxxx] = np.sum(chclanku[1][xxxx])
        
    for x in range(len(w1)):
        for y in range(len(w1[x])):
            chclanku[0][x][y] = chneuronu[0][x] * (w1[x][y]/np.sum(w1[x]))


    alfa = 0.2
    for xxx, hodnota in enumerate(w3):
        for yyy, hodnotay in enumerate(hodnota):
            w3[xxx][yyy] = w3[xxx][yyy] - alfa * (-chclanku[2][xxx][yyy]
                                                *output_layer[xxx] * (1 - output_layer[xxx]) *hidden_layer2[xxx])
    for xxx, hodnota in enumerate(w2):
        for yyy, hodnotay in enumerate(hodnota):
            w2[xxx][yyy] = w2[xxx][yyy] - alfa * (-chclanku[1][xxx][yyy]
                                                *hidden_layer2[xxx] * (1 - hidden_layer2[xxx]) *hidden_layer1[xxx])
    for xxx, hodnota in enumerate(w1):
        for yyy, hodnotay in enumerate(hodnota):
            w1[xxx][yyy] = w1[xxx][yyy] - alfa * (-chclanku[0][xxx][yyy]
                                                *hidden_layer1[xxx] * (1 - hidden_layer1[xxx]) *data[0][xxx])
    print(labels[asd], "->", output_layer)

            

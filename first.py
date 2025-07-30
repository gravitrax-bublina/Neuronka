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
learningrate = 1/3
w1 = np.random.uniform(0, 1/((28*28)**0.5), (256, 784))  # 16 hidden neurons, 784 inputs
print(w1)
w2 = np.random.uniform(0, 1/((256)**0.5), (32, 256)) 
w3 = np.random.uniform(0, 1/((32)**0.5), (10, 32)) 
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
def vyslednakalkulace(v1,v2,v3, l1,l2,l3,l4,e2,e3,e4,lr):
    e = [e2, e3, e4]
    v = [v1, v2, v3]
    l = [l1, l2, l3, l4]
    zmena = [np.zeros_like(v1),
        np.zeros_like(v2),
        np.zeros_like(v3)]
    for x in range(len(zmena)):
        zmena[x] = np.dot(lr*e[x]*l[x]*(1-l[x]),l[x-1].T)

# Neural net    work weights
   # 10 output neurons, 16 hidden neurons
# Test input values
for asd in range(len(data)):
        
    # Forward propagation
    hidden_layer1 = sigmoid(np.dot(w1, data[asd]))  # Using the first label as input
    hidden_layer2 = sigmoid(np.dot(w2, hidden_layer1))
    output_layer = sigmoid(np.dot(w3, hidden_layer2))

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
    for x in range(len(chclanku)):
        if x==0:
            chclanku[x] = np.dot(w3.T, er[0])
        else:
            chclanku[x] = np.dot(w3.T, chclanku[x-1])


    vyslednakalkulace(w1,w2,w3,data[asd], hidden_layer1, hidden_layer2, output_layer, chclanku[0], chclanku[1], er[0],learningrate)


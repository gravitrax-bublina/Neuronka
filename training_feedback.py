import numpy as np 
import csv
import json

# Load MNIST data from CSV file
# Assuming the CSV file is formatted with each row as an image and the first column as the label
# For example, the first row might look like: [label, pixel1, pixel2    , ..., pixel784]   
def uloz(w):
    np.savez("Neuronka/OBROVSKA_NEURONOVA_SIT222.npz", *w)

def load():
    data = np.load("Neuronka/OBROVSKA_NEURONOVA_SIT222.npz")
    w = [data[key] for key in data.files]
    return w

## input data loading
def berdata(umisteni):
    with open(umisteni, "r") as f:
        dataa = []
        labels = []
        # scratch first line
        
        
        for line in f:
            
            values = line.split(",")
            label = int(values[0])               # first value is label
            pixels = [int(v) / 255 for v in values[1:]]  # rest is pixel data

            labels.append(label)
            dataa.append(pixels)

        
        dataa = np.array(dataa) # ??
        return [labels, dataa]

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

## Network parameters
node_sizes = [784, 512, 256, 128, 10]
learningrate = 0.01
w = []
for i in range(len(node_sizes) - 1):
    w.append(np.random.normal(0, 1/((node_sizes[i])**0.5), (node_sizes[i+1], node_sizes[i])))   
def vyslednakalkulace(layers, errors, lr):
    e = [np.array(x, ndmin=2) for x in errors]
    l = [np.array(x, ndmin=2) for x in layers]

    pocet_vrstev = len(layers) - 1
    zmena = [0] * pocet_vrstev
    for x in range(pocet_vrstev):
        zmena[x] = lr*np.dot((e[x]*l[x+1]*(1-l[x+1])).T, l[x])
    
    return zmena

def jakecislokdyznevim(ol):
    max_index = np.argmax(ol)
    nejvyssi = ol[max_index]
    return [max_index]

def jakecislo(label, ol, asd, n):
    max_index = np.argmax(ol)
    nejvyssi = ol[max_index]
    if label == max_index:
        correctness = True
    else:
        correctness = False
        if asd%n == 0:
            print("asd: ", asd, "label: ", label, "max_index: ", max_index, "nejvyssi: ", nejvyssi, "correctness: ", correctness)

    return [correctness, nejvyssi]
def forward_propagation(input_data, w): 
    # Forward propagation
    #print(input_data.shape)
    input_np = np.array(input_data, ndmin=2)
    #print(w[0].shape, input_np.shape)
    hidden_layers = [sigmoid(np.dot(w[0], input_data))]  # Using the first label as input
    for w_i in range(1, len(w)-1):
        hidden_layers.append(sigmoid(np.dot(w[w_i], hidden_layers[-1])))
    output_layer = sigmoid(np.dot(w[-1], hidden_layers[-1]))
    return hidden_layers, output_layer

# Test input values

def train(umisteni):
    global w

    uspesnosti = []
    labels, data = berdata(umisteni) 
    for asd in range(len(data)):

        # Forward propagation
        hidden_layers, output_layer = forward_propagation(data[asd], w)
        # Backward propagation
        er = np.zeros(len(output_layer))
        vec_label = []
        
        for i in range(len(output_layer)):
            if i == labels[asd]:
                vec_label.append(1)
            else:
                vec_label.append(0)
            
            er[i] = vec_label[i] - output_layer[i]

        chclanku = [np.zeros(ns) for ns in node_sizes[1:-1]]
        for x in range(len(chclanku)):
            if x==0:
                chclanku[x] = np.dot(w[-1].T, er)
            else:
                chclanku[x] = np.dot(w[-1-x].T, chclanku[x-1])
        chclanku.reverse()

        deltas = vyslednakalkulace(
            [data[asd]] + hidden_layers + [output_layer],  
            chclanku + [er], 
            learningrate)
        
        for x in range(len(w)):
            w[x] += deltas[x]
        
        n=1000
        uspesnosti.append(jakecislo(labels[asd], output_layer, asd, n))
        
        if(asd%n == 0):
            print("output",output_layer, labels[asd], uspesnosti[asd], n)
    
    return uspesnosti
        

if __name__ == "__main__":
    # train the network
    train("Neuronka/mnist_train.csv")

    # test the network
    print("test")
    vysledky = train("Neuronka/mnist_test.csv")
    uspesnost = 0
    probabilitaus = 0
    for v in vysledky:
        if v[0]:
            uspesnost += 1
        
        probabilitaus += v[1] 
        
    probabilitaus = probabilitaus / len(vysledky)
    uspesnost = uspesnost / len(vysledky)
    print ("certainty: ", probabilitaus, "uspesnost: ", uspesnost)

    uloz(w)



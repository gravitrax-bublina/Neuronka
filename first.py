import numpy as np 
import csv
import json

# Load MNIST data from CSV file
# Assuming the CSV file is formatted with each row as an image and the first column as the label
# For example, the first row might look like: [label, pixel1, pixel2    , ..., pixel784]   
def uloz(w):
    np.savez("OBROVSKA_NEURONOVA_SIT2.npz", *w)
def load():
    data = np.load("OBROVSKA_NEURONOVA_SIT2.npz")
    w = [data[key] for key in data.files]
    return w
def berdata(umisteni):
    with open(umisteni, "r") as f:
        dataa = []
        labels = []
        # Read the CSV file

        for line in f:
            values = line.split(",")
            label = int(values[0])               # first value is label
            pixels = [int(v) for v in values[1:]]  # rest is pixel data

            labels.append(label)
            dataa.append(pixels)
        dataa = np.array(dataa)
        dataa = dataa /265
        return([labels, dataa])
learningrate = 1/10
#data = [data[0]]
#w1 = np.random.normal(0, 1/((28*28)**0.5), (256, 784))  # 16 hidden neurons, 784 inputs
w2 = np.random.normal(0, 1/((28*28)**0.5), (100, 784)) 
w3 = np.random.normal(0, 1/((32)**0.5), (10, 100)) 
w = [w2, w3]



def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
def vyslednakalkulace(v1,v2,
                      #v3,
                      l1,l2,l3,
                      #l4,
                       e2,e3,
                       #e4,
                       #
                       lr):
    e = [np.array(x, ndmin=2) for x in [e2, e3, 
                                        #e4
                                        ]]
    v = [np.array(x, ndmin=2) for x in [v1, v2, 
                                        #v3
                                        ]]
    l = [np.array(x, ndmin=2) for x in [l1, l2, l3, 
                                        #l4
                                        ]]
    
    zmena = [np.zeros_like(v1),
        np.zeros_like(v2),]
        #np.zeros_like(v3)]
    for x in range(len(zmena)):
        #print(x, len(e[x]),len(l[x+1]))
        #print(e[x]*l[x+1]*(1-l[x+1]))
        #print(l[x].T)
        zmena[x] = lr*np.dot((e[x]*l[x+1]*(1-l[x+1])).T, l[x])
        #print("zmena: ")
        #print(len(zmena[x]), len(zmena[x][0]))
    return zmena
def jakecislo(label,ol,asd,n):
    nejvyssi = 0
    #print(ol)
    for x in ol:
        if x > nejvyssi:
            nejvyssi = x
    #print(nejvyssi)
    
    if nejvyssi > ol[label]:
        if(asd%n == 0): 
            print(nejvyssi, ol[label], label, "fc")
        return(False, nejvyssi)
    else:
        return(True, nejvyssi)
        
    

# Neural net    work weights
   # 10 output neurons, 16 hidden neurons
# Test input values
uspesnosti = []
def rantimee(analyse, umisteni):
    liist=berdata(umisteni)
    labels = liist[0]
    
    data = liist[1]
    for asd in range(len(data)):
            
        # Forward propagation
        hidden_layer1 = sigmoid(np.dot(w[0], data[asd]))  # Using the first label as input
        #hidden_layer2 = sigmoid(np.dot(w[1], hidden_layer1))
        output_layer = sigmoid(np.dot(w[1], hidden_layer1))

        #print("Output layer (after w3):", output_layer)
        er = []
        nextgenlabels = []
        for x in range(1):
            er.append(np.zeros(len(output_layer)))  # dáváme desítky nul (jen jedna chic hic hichi)
            nextgenlabels = []
            
            for i in range(len(output_layer)):
                if i == labels[asd]:
                    nextgenlabels.append(1)
                else:
                    nextgenlabels.append(0)
                
                if x == 0:
                    er[x][i] = (nextgenlabels[i] - output_layer[i])
        #print(nextgenlabels)



        chclanku = [
            np.zeros_like(w[0]),
            np.zeros_like(w[1])#,
            #np.zeros_like(w[2])
        ]
        chneuronu = [hidden_layer1]#, hidden_layer2]
        for x in range(len(chclanku)):
            if x==0:
                chclanku[x] = np.dot(w[1].T, er[0])
            else:
                chclanku[x] = np.dot(w[1-x].T, chclanku[x-1])

        vysledky = [np.zeros_like(w[0]),
            np.zeros_like(w[1]),]
            #np.zeros_like(w[2])]
        deltas = vyslednakalkulace(
            w[0],w[1],#w[2],
            data[asd], 
            hidden_layer1, #hidden_layer2, 
            output_layer,
            #chclanku[1], 
            chclanku[0], er[0], learningrate)
        for x in range(len(w)):
            w[x] += deltas[x]
        n=1000
        uspesnosti.append([jakecislo(labels[asd], output_layer,asd, n)])
        
        if(asd%n == 0):
            
            print("output",output_layer, labels[asd], uspesnosti[asd], n)
    return(uspesnosti)
        
for x in range(2):

    if x==0:
        dd = rantimee(True, "mnist_train.csv")
        uspesnost = 0
        probabilitaus = 0
        for x in range(len(dd)):
            if dd[x][0][0] == True:
                uspesnost = uspesnost + 1
            
            probabilitaus = probabilitaus + dd[x][0][1] 
            
        probabilitaus = probabilitaus / len(dd)
        uspesnost = uspesnost / len(dd)
        print ("certainty: ", probabilitaus, "uspesnost: ", uspesnost)
    else:
        print("test")
        vysledkyy = rantimee(True, "mnist_test.csv")
        uspesnost = 0
        probabilitaus = 0
        for x in range(len(vysledkyy)):
            if vysledkyy[x][0][0] == True:
                uspesnost = uspesnost + 1
            
            probabilitaus = probabilitaus + vysledkyy[x][0][1] 
            
        probabilitaus = probabilitaus / len(vysledkyy)
        uspesnost = uspesnost / len(vysledkyy)
        print ("certainty: ", probabilitaus, "uspesnost: ", uspesnost)
        uloz(w)



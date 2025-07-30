    for x in range(len(w3)):
        sumaw3x=np.sum(w3[x])
        for y in range(len(w3[x])):
            chclanku[2][x][y] = (er[0][x]*w3[x][y])/sumaw3x #x u chclanku je jako x u w3 tedy pro kazdej neuron

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


    alfa = 0.5
    for xxx, hodnota in enumerate(w3):
        for yyy, hodnotay in enumerate(hodnota):
            w3[xxx][yyy] = w3[xxx][yyy] - alfa * (-chclanku[2][xxx][yyy]
                                                *output_layer[xxx] * (1 - output_layer[xxx]) *hidden_layer2[yyy])
    for xxx, hodnota in enumerate(w2):
        for yyy, hodnotay in enumerate(hodnota):
            w2[xxx][yyy] = w2[xxx][yyy] - alfa * (-chclanku[1][xxx][yyy]
                                                *hidden_layer2[xxx] * (1 - hidden_layer2[xxx]) *hidden_layer1[yyy])
    for xxx, hodnota in enumerate(w1):
        for yyy, hodnotay in enumerate(hodnota):
            w1[xxx][yyy] = w1[xxx][yyy] - alfa * (-chclanku[0][xxx][yyy]
                                                *hidden_layer1[xxx] * (1 - hidden_layer1[xxx]) *data[asd][yyy])
    print(labels[asd], "->", output_layer)

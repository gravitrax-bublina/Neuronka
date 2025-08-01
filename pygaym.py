import numpy as np, pygame, time
from training_feedback import train, vyslednakalkulace, jakecislo, uloz, load, forward_propagation, sigmoid, jakecislokdyznevim, berdata
import sys
sys.path.append(r"C:/Users/gravi/OneDrive/Plocha/protab/neural_network")

from csv_to_image import csv_to_image
outputovylist = []
node_sizes = [784, 512, 256, 10]
pygame.font.init()
font = pygame.font.SysFont("freesansbold", 100)
mezera = 1
w = load()
pygame.init()
kolecka = []
timer = pygame.time.Clock()
fps = 120
height = width = 28*32
screen = pygame.display.set_mode((width, height))
kokolsize = 32
pygame.display.set_caption("Neuronka")
barva = "bila"
def komprese(areje):
    outputovylist = ""
    for y in range(len(areje)):
        for x in range(len(areje[y])):
            if x%32 ==0 and y%32==0:
                window = areje[y:y+32, x:x+32]  # shape (32, 32, 3)
                average_rgb = np.mean(window, axis=(0, 1))  # shape (3,), avg R, G, B
                avg_rgb_value = np.mean(average_rgb)  # scalar: (R + G + B)/3
                outputovylist += str(255-round(avg_rgb_value)) + ","

    with open("C:/Users/gravi/OneDrive/Plocha/protab/neural_network/output.csv", 'w') as f:
        print("outputovy list: " +outputovylist)
        f.write("-1," + outputovylist[:-1])  # Remove the last comma
    pygame.font.init()
    pygame.init()
    
    ofcislo = jakecislokdyznevim(forward_propagation(berdata("C:/Users/gravi/OneDrive/Plocha/protab/neural_network/output.csv")[1].T, w)[1])
    screen = pygame.display.set_mode((600, 400))
    text_surface = font.render(str(ofcislo), True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(300, 200))
    screen.fill((0, 0, 0))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    pygame.time.delay(1000)
    pygame.quit()
    return
if barva == "bila":
    barva = (255, 255, 255)
    bcerna = (0, 0, 0)
else:
    barva = (0, 0, 0)
    bcerna = (255, 255, 255)
last_pos = (0, 0)
run = True
def kresli(koleckaa):
    global kolecka   
    global latn
    global mouse
    
    if not koleckaa:
        return
    if latn:
        
        deltax = koleckaa[-1][1][0] - last_pos[0]
        deltay = koleckaa[-1][1][1] - last_pos[1]
        distance = (deltax**2 + deltay**2)**0.5
        print(distance, deltax)
        for vzdalenost in range(int(distance)):
            x_pos = int(last_pos[0] + (deltax * vzdalenost / distance))
            y_pos = int(last_pos[1] + (deltay * vzdalenost / distance))
            pygame.draw.circle(screen, koleckaa[-1][0], (x_pos, y_pos), koleckaa[-1][2])
            kolecka.append((koleckaa[-1][0], (x_pos, y_pos), koleckaa[-1][2], last_pos))
            print((koleckaa[-1][0], (x_pos, y_pos), koleckaa[-1][2], last_pos))
    else:
        print(latn)
    for x in koleckaa:
        pygame.draw.circle(screen, x[0], x[1], x[2])
howmanyclicks = 0
while run:
    timer.tick(fps)
    screen.fill(barva)
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False
    mouse = pygame.mouse.get_pos()
    latn = pygame.mouse.get_pressed()[0]
    if latn:
            kolecka.append((bcerna, mouse, kokolsize, last_pos))
    patn = pygame.mouse.get_pressed()[1]
    kresli(kolecka)
    pygame.display.flip()
    
    if not latn:
        if patn:
            kolecka.append((barva, mouse, kokolsize))
        pygame.draw.circle(screen, bcerna, mouse, kokolsize)
        if howmanyclicks > 0:
            
            run = False
            array = pygame.surfarray.array3d(screen)

            # Create a boolean mask where pixels are NOT white
            non_white_mask = np.any(array != [255, 255, 255], axis=2)

            # Count the number of non-white pixels
            non_white_count = np.count_nonzero(non_white_mask)

            print("Number of non-white pixels:", non_white_count)

            array = array.transpose(1, 0, 2)
            komprese(array)

            time.sleep(mezera)
            pygame.quit()
    else:
        if latn:
            kolecka.append((bcerna, mouse, kokolsize, last_pos))
            howmanyclicks += 1
    
    
    #for x in kolecka:
    #    pygame.draw.circle(screen, x[0], x[1], x[2])
    last_pos = mouse

from csv_to_image import csv_to_image
csv_to_image()
pygame.quit()
#output_layer = forward_propagation(np.zeros(node_sizes[0]), w)[1]


#Harita hücrelerden oluşur. Hücre değerleri:

#0: boş
#1: dolu
#-1: bilinmiyor

#gmapping, hector_slam bu yapıyı üretir:

rosrun map_server map_saver -f my_map

#Python ile görselleştirme:

import matplotlib.pyplot as plt
import numpy as np

grid = np.random.choice([-1, 0, 1], size=(20, 20), p=[0.1, 0.7, 0.2])
plt.imshow(grid, cmap='gray')
plt.title("Occupancy Grid Map")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import os


for file in os.listdir(r"."):
    if "csv" in file:
        data = np.loadtxt(file, dtype = float, delimiter=',', skiprows= 1)                    
        derivative = np.gradient(data[:,1], data[:,0])
        x = []
        y = []

        for i in range(len(derivative)):
            if np.absolute((derivative[i]-derivative[i-1])) < 1e6:
                x.append(data[i,0])
                y.append(derivative[i])
        plt.plot(x,y) 

        print(file)
        print(x[np.argmax(y)])

plt.show()
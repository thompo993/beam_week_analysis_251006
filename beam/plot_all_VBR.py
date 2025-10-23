import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import personalpaths

FOLDER_PATH = personalpaths.VBR_PATH#r"data" #folderpath goes here, ideally nothing else in the folder

def line(m,q,x):
    return m*x+q

x = np.arange(62)
col =["r", "g", "b", "k"]
vbr = []
lab = []
k = 0

for file in os.listdir(FOLDER_PATH):
    lab.append(file[-6:-4])
    file_path = os.path.join(FOLDER_PATH , file)
    df = np.loadtxt(file_path,delimiter=",")
    for i in range(int(len(df)/2)):
        if df[1+2*i] > 30: #reject lines that were not fitted right. best course of action would be to find those and check why
            plt.plot(x, line(df[0+2*i], df[1+2*i],x), color = col[k], linewidth = 0.5 )

            vbr.append(df[1+2*i])
    k+=1

#create legend and plot
legend_elements = [Line2D([0], [0], color=col[0], lw=4, label=lab[0]),
                   Line2D([0], [0], color=col[1], lw=4, label=lab[1]),
                   Line2D([0], [0], color=col[2], lw=4, label=lab[2]),
                   Line2D([0], [0], color=col[3], lw=4, label=lab[3])]
plt.title("Vbr")
plt.xlim(0,None)
plt.legend(handles=legend_elements)
plt.xlabel("PE")
plt.ylabel("V")
plt.grid()
plt.show()

#same thing, just a more zoomed in version of the code
k = 0
labz = []
xzoom = np.linspace(20,45)

for file in os.listdir(FOLDER_PATH):
    labz.append(file[-6:-4])
    file_path = os.path.join(FOLDER_PATH , file)
    df = np.loadtxt(file_path,delimiter=",")
    for i in range(int(len(df)/2)):
        if df[1+2*i] > 30: #reject lines that were not fitted right
            plt.plot(xzoom, line(df[0+2*i], df[1+2*i],xzoom), color = col[k], linewidth = 0.5 )
    k+=1

legend_elements = [Line2D([0], [0], color=col[0], lw=4, label=labz[0]),
                   Line2D([0], [0], color=col[1], lw=4, label=labz[1]),
                   Line2D([0], [0], color=col[2], lw=4, label=labz[2]),
                   Line2D([0], [0], color=col[3], lw=4, label=labz[3])]

# Create the figure
plt.title("Zoomed")
plt.legend(handles=legend_elements)
plt.xlabel("PE")
plt.ylabel("V")
plt.grid()
plt.show()

#plots histogram 
plt.hist(vbr,15)
plt.show()


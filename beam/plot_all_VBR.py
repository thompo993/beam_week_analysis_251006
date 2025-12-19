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

plt.figure(figsize=[14,10])

for file in os.listdir(FOLDER_PATH):
    lab.append(file[-6:-4])
    file_path = os.path.join(FOLDER_PATH , file)
    if "csv" in file:    
        df = np.loadtxt(file_path,delimiter=",")
        for i in range(int(len(df)/2)):
            if df[1+2*i] < 53: #reject lines that were not fitted right. best course of action would be to find those and check why
                plt.plot(x, line(df[0+2*i], df[1+2*i],x), color = col[k], linewidth = 0.5 )

                vbr.append(df[1+2*i])
        k+=1

#create legend and plot
legend_elements = [Line2D([0], [0], color=col[0], lw=4, label=lab[0]),
                   Line2D([0], [0], color=col[1], lw=4, label=lab[1]),
                   Line2D([0], [0], color=col[2], lw=4, label=lab[2]),
                   Line2D([0], [0], color=col[3], lw=4, label=lab[3])]
plt.title("Linear fit of gain profile",fontsize =20)
plt.xlim(0,None)
plt.legend(handles=legend_elements)
plt.xlabel("\u0394LSB per photo-electron")
plt.ylabel("Bias voltage [V]")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "vbr"))
plt.show()

#same thing, just a more zoomed in version of the code
k = 0
labz = []
xzoom = np.linspace(20,45)


plt.figure(figsize=[14,10])

for file in os.listdir(FOLDER_PATH):
    labz.append(file[-6:-4])
    file_path = os.path.join(FOLDER_PATH , file)
    if "csv" in file:
        df = np.loadtxt(file_path,delimiter=",")
        for i in range(int(len(df)/2)):
            if df[1+2*i] < 53: #reject lines that were not fitted right
                plt.plot(xzoom, line(df[0+2*i], df[1+2*i],xzoom), color = col[k], linewidth = 0.5 )
        k+=1

legend_elements = [Line2D([0], [0], color=col[0], lw=4, label=labz[0]),
                   Line2D([0], [0], color=col[1], lw=4, label=labz[1]),
                   Line2D([0], [0], color=col[2], lw=4, label=labz[2]),
                   Line2D([0], [0], color=col[3], lw=4, label=labz[3])]

# Create the figure
plt.title("Linear fit of gain profile (zoomed in)", fontsize = 20)
plt.legend(handles=legend_elements)
plt.xlabel("\u0394LSB per photo-electron")
plt.ylabel("Bias voltage [V]")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "vbr_zoom"))
plt.show()

#plots histogram 

plt.figure(figsize=[14,10])
plt.hist(vbr,15)
plt.title("Histogram of break-down voltage", fontsize = 20)
plt.xlabel("Break-down voltage")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FOLDER_PATH, "vbr_hist"))
plt.show()


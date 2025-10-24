import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import personalpaths
from matplotlib.lines import Line2D

FOLDER_PATH = personalpaths.FOLDER_PATH_BARREL # if not using personalpaths file, use r"filepath"
xlim = [0, 3500]
ylim = [0, 2]
size = [14,10]
PE = 25
alpha = 0.5


fig, ax = plt.subplots(figsize=size)


for file in os.listdir(FOLDER_PATH):
    if "csv" in file:
        file_path = os.path.join(FOLDER_PATH, file)
        if "AA" in file:
            df = np.loadtxt(file_path, delimiter=",", dtype=float)

            channel = 3
            y = df[:,channel]
            x = range(len(df))
            # x = [i/3 for i in x]
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, alpha = alpha, label = "Inner ring")

for file in os.listdir(FOLDER_PATH):
    if "csv" in file:
        file_path = os.path.join(FOLDER_PATH, file)
        if "BB" in file:        
            df = np.loadtxt(file_path, delimiter=",", dtype=float)

            channel = 1
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, alpha = alpha, label = "Ring 5")

            channel = 5
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, alpha = alpha, label = "Ring 6")
        
for file in os.listdir(FOLDER_PATH):
    if "csv" in file:
        file_path = os.path.join(FOLDER_PATH, file)
        if "C_" in file:       
            df = np.loadtxt(file_path, delimiter=",", dtype=float)

            channel = 1
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, alpha = alpha, label = "Ring 7")

            channel = 7
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, alpha = alpha, label = "Ring 8")
        
        
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel("LSB")
secx = ax.secondary_xaxis('top')
secx.set_xticklabels(["{:.0f}".format(x/PE) for x in ax.get_xticks()])
secx.set_xlabel("Photons")
ax.set_ylabel("Counts [Area Normalised]")
ax.grid()

ax.set_title("Characteristic Pulse Height Spectra for each ring", fontsize = 15)

ax.legend()

ax.legend()
fig.tight_layout()

filename = "ring"
plt.savefig(os.path.join(FOLDER_PATH,filename))
plt.show() 

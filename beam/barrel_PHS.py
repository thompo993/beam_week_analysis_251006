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
ylim = [0, 4]
size = [14,10]
PE = 25


fig, ax = plt.subplots(figsize=size)


for file in os.listdir(FOLDER_PATH):

    if "csv" in file:
        file_path = os.path.join(FOLDER_PATH, file)
        df = np.loadtxt(file_path, delimiter=",", dtype=float)
        if "AA" in file:
            channel = 3
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, label = "Inner ring")

        if "BB" in file:        
            channel = 1
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, label = "Ring 5")

            channel = 5
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, label = "Ring 6")
        
        if "C_" in file:
            channel = 1
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, label = "Ring 7")

            channel = 7
            y = df[:,channel]
            x = range(len(df))
            norm = np.sum(y)/1000
            ax.plot(x,y/norm, label = "Ring 8")
        
        
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel("LSB")
secx = ax.secondary_xaxis('top')
secx.set_xticklabels(["{:.0f}".format(x/PE) for x in ax.get_xticks()])
secx.set_xlabel("Photons")
ax.set_ylabel("Counts [Area Normalised]")
ax.grid()

ax.set_title("Rings PHS")
ax.legend()
fig.tight_layout()

filename = "ring"
# plt.savefig(os.path.join(FOLDER_PATH,filename))
plt.show() 

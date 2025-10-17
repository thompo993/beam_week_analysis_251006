import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

FOLDER_PATH = r"20251011_130933" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
description = "" #add what you want the title to say, 
if description != "":
    description = " - " + description

xlim = [None, None]
ylim = [None, 1000]
size = [10,8]

def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df, label = folder[-10:], linewidth = 0.5)
    plt.title("Channel %s"%num + description)
    plt.yscale(LOG)
    plt.xlim(xlim)
    plt.ylim(ylim)
    leg = plt.legend()

    for legobj in leg.legend_handles:
        legobj.set_linewidth(4.0)
    
    plt.grid()
    if SAVE_PNG:
        plt.savefig(os.path.join(FOLDER_PATH,"CH_%s"%num))
        plt.close()
    else: plt.show() 



for i in range(8):
    plot_channels(i)

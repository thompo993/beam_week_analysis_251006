import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import personalpaths

FOLDER_PATH = personalpaths.MISMATCHED_PATH#r"folderpath" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
description = "" #add what you want the title to say, if nothing leave a "" 
if description != "":
    description = " - " + description

xlim = [None, None]
ylim = [-10, 1000] #choose appropriate limits, only applies to linear plots
size = [10,8]

def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH): #looks into all folders 
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                #plot for each file
                plt.plot(x, df, label = folder[-10:], linewidth = 0.5)
    plt.title("Channel %s"%num + description)
    plt.yscale(LOG)
    if LOG == 'linear':    
        plt.ylim(ylim)
    plt.xlim(xlim)
    leg = plt.legend()

    for legobj in leg.legend_handles: #increase size of lines in legend to show colors better
        legobj.set_linewidth(4.0)
    
    plt.grid()
    if SAVE_PNG:#if true save, if false show
        plt.savefig(os.path.join(FOLDER_PATH,"CH_%s"%num+LOG))
        plt.close()
    else: plt.show() 



for i in range(8): #do this for all channels 
    plot_channels(i)

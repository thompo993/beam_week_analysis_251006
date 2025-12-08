import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import personalpaths as personalpaths 
plt.rcParams.update({'font.size': 20})


FOLDER_PATH = personalpaths.MISMATCHED_PATH#r"folderpath" #folderpath goes here, ideally nothing else in the folder

SAVE_PATH = os.path.join(FOLDER_PATH, "figures")
os.makedirs(SAVE_PATH,exist_ok=True)

SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
PE = 70
description = "Module C" #add what you want the title to say, if nothing leave a "" 
if description != "":
    description = " - " + description

xlim = [0, 4095]
ylim = [0, 1] #choose appropriate limits, only applies to linear plots
size = [14,10]
rebinning = True

def rebin(x,n): 
    rebin = []
    # tmp = [x[i] for i in range(len(x))]
    tmp = np.array(x)

    for k in range(2*n):
        tmp = np.append(tmp,0)
 
    for i in range(len(x)):
        bin = tmp[i-n:i+n]
        rebin.append(np.average(bin))
    return rebin

def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH): #looks into all folders 
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file and "amplitude" in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                norm = np.sum(np.array(df))/1000
                if rebinning:
                    df = rebin(df, 8)
                df = df/norm
                x = range(len(df))
                #plot for each file
                plt.plot(x, df, label = "Right \u0394LSB/PE = " + folder[-3:])
    plt.title("Mismatched scan, Left \u0394LSB/PE = %s - Channel %s"%(PE,num) + description)
    plt.yscale(LOG)
    if LOG == 'linear':    
        plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("LSB" )
    plt.ylabel("Counts [Area Normalised]")

    leg = plt.legend(fontsize = 14)

    for legobj in leg.legend_handles: #increase size of lines in legend to show colors better
        legobj.set_linewidth(4.0)
    
    plt.tight_layout()
    plt.grid()
    if SAVE_PNG:#if true save, if false show
        plt.savefig(os.path.join(SAVE_PATH,"CH_%s"%num+LOG))
        plt.close()
    else: plt.show() 



for i in range(8): #do this for all channels 
    plot_channels(i)

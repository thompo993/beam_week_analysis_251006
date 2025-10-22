import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

FOLDER_PATH = r"C:\Users\xph93786\Desktop\first_plots"#folderpath goes here


SAVE_PNG = False
LOG = 'linear' #options: 'log', 'linear'
channel = 1

xlim = [None, 3000]
ylim = [0, None]
size = [10,8]

#parameters for peak finding 
threshold = 230
p_height = 0.00088



def smooth_data(y, window=51, poly=3):
    """Apply Savitzky-Golay smoothing filter"""
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)


plt.figure(figsize=size)
for file in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file)
    df = np.loadtxt(file_path, delimiter=",", dtype=float)
    if df.ndim > 1: 
        df = df[:,channel]
        df.flatten()
    x = range(len(df))
    norm = np.sum(df)

    y = smooth_data(df/norm)
    y = smooth_data(y)    
    y = smooth_data(y)    
    y = smooth_data(y)
    peaks, _ = find_peaks(y, height = p_height)

    pk = [i for i in peaks if i>threshold]


    plt.plot(x, df/norm, label = file, alpha = 0.5)
    if len(pk) != 0:
        plt.scatter(pk, df[pk]/norm, color ="k")

    if len(pk)!=1:
        print(len(pk))
        print(file)
        

plt.title("CH %s" %channel)
plt.yscale(LOG)
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.grid()

plt.show() 


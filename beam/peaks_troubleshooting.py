import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

threshold = 230
peak_pos = 500

FOLDER_PATH = r"C:\Users\xph93786\Desktop\first_plots\amplitude_ch1_vertical.csv"#folderpath goes here


SAVE_PNG = False
LOG = 'linear' #options: 'log', 'linear'
channel = 1

xlim = [100, 1000]
ylim = [None, None]
size = [10,8]



def smooth_data(y, window=51, poly=3):
    """Apply Savitzky-Golay smoothing filter"""
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)


plt.figure(figsize=size)

file_path = FOLDER_PATH
df = np.loadtxt(file_path, delimiter=",", dtype=float)
norm = np.sum(df)
df = -1*df
if df.ndim > 1: 
    df = df[:,channel]
    df.flatten()
x = range(len(df))

y = smooth_data(df/norm)
y = smooth_data(y)    
y = smooth_data(y)    
y = smooth_data(y)
peaks, _ = find_peaks(y, height = -0.00088)

print(peaks)
pk = [i for i in peaks if i<peak_pos and i >threshold ]
print(pk)



plt.plot(x, df/norm)
plt.plot(x,y)
if len(pk) != 0:
    plt.scatter(pk, df[pk]/norm, color ='r')
    print(len(pk))

plt.title("CH %s" %channel)
plt.yscale(LOG)
plt.xlim(xlim)
plt.ylim(ylim)
plt.grid()

plt.show() 


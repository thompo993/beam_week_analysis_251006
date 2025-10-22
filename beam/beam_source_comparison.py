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

xlim = [None, 1000]
ylim = [0, None]
size = [10,8]

#parameters for peak finding 
threshold = 230
p_height = 0.88
p_to_v_diff = 0.15



def smooth_data(y, window=51, poly=3):
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)

def peak_finder(data):
    y = smooth_data(data)
    y = smooth_data(y)    
    y = smooth_data(y)    
    y = smooth_data(y)
    peaks, _ = find_peaks(y, height = p_height)
    pk = [i for i in peaks if i>threshold]

    if len(pk) != 0:
        plt.scatter(pk, data[pk], color ="k")

    if len(pk)!=1:
        print(len(pk))
        print(file)
    return pk

def valley_finder(data, m_peak, p_h):
    y = smooth_data(-data)
    y = smooth_data(y)    
    y = smooth_data(y)    
    y = smooth_data(y)
    valley, _ = find_peaks(y, height = float(-(p_h-p_to_v_diff)))
    vl = [i for i in valley if i>threshold and i<m_peak]


    if len(vl)!=0:
        if len(vl)!=1:
            print(len(vl))
            print(file)
            vl = int(np.average(vl))
        plt.scatter(vl, data[vl], color ="r")

    return vl

        

plt.figure(figsize=size)
for file in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file)
    df = np.loadtxt(file_path, delimiter=",", dtype=float)
    if df.ndim > 1: 
        df = df[:,channel]
        df.flatten()
    x = range(len(df))
    norm = np.sum(df)/1000

    plt.plot(x, df/norm, label = file, alpha = 0.5)
    
    peak = peak_finder(df/norm)
    peak_coordinates = [peak,df[peak]/norm]   
    val = valley_finder(df/norm, peak,df[peak]/norm )
    valley_coordinates = [val,df[val]/norm]
    print(peak_coordinates)
    print(valley_coordinates)
   

    plt.title("CH %s" %channel)
    plt.yscale(LOG)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.grid()

    plt.show() 


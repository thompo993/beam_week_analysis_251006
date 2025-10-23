import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

FOLDER_PATH = r"C:\Users\xph93786\Desktop\first_plots"#folderpath goes here

LOG = 'linear' #options: 'log', 'linear'
channel = 5

xlim = [0, 1500]
ylim = [0, 4]
size = [10,8]

#parameters for peak finding 
threshold = 250
p_height = 0.76
p_to_v_diff = 0.15
hwhm_ave = 0.01 #averages +- 1% of half height 
ran = 250

SHOW_GAUSS = True

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def smooth_data(y, window=70, poly=4):
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)

def peak_finder(data):
    y = smooth_data(data)
    plt.plot(y)
    peaks, _ = find_peaks(y, height = p_height)
    for i in range(len(peaks)):
        if peaks[i]-peaks[i-1] > ran:
            cut = i
    peaks = peaks[i:]
    pk = [i for i in peaks if i>threshold]
    x = np.arange(len(y))

    peak_guess = np.average(pk)
    min_range = int(peak_guess - ran)
    if min_range < threshold:
        min_range = threshold
    max_range = int(peak_guess + ran)
    y = y[min_range:max_range]  
    x = x[min_range:max_range]
    popt, _ = curve_fit(gaussian,x,y,p0 = [p_height, peak_guess,peak_guess])
   
    if SHOW_GAUSS:
        plt.plot(x, gaussian(x,*popt))
    plt.scatter(popt[1], gaussian(popt[1],*popt), color = "r")

    return [popt[1], gaussian(popt[1],*popt)]

def valley_finder(data, m_peak, p_h):
    y = smooth_data(-data)
    y = smooth_data(y)    
    y = smooth_data(y)    
    y = smooth_data(y)
    valley, _ = find_peaks(y, height = float(-(p_h-p_to_v_diff)))
    vl = [i for i in valley if i>threshold and i<m_peak]


    if len(vl)!=0:
        if len(vl)>0:
            vl = int(np.average(vl))
        plt.scatter(vl, data[vl], color ="r")
    return vl

def HWHM_right(coordinates, data):
    data=smooth_data(data)
    x = []
    y = []
    half = coordinates[1]/2
    for k in range(int(coordinates[0]), len(data)):
        if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
            x.append(k)
            y.append(data[k])
    plt.scatter(np.average(x), np.average(y), color = 'g')
    return np.average(x)-coordinates[0]
    
f = open(os.path.join(FOLDER_PATH, 'log.txt'), 'w')

f.write("Analysis parameters:\n")
f.write("threshold: %s\n" %threshold)
f.write("p_height: %s\n" %p_height)
f.write("p_to_v_diff: %s\n" %p_to_v_diff)
f.write("hwhm_ave: %s\n\n" %hwhm_ave)


plt.figure(figsize=size)
for file in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file)
    if "csv" not in file:
        continue
    else:
        df = np.loadtxt(file_path, delimiter=",", dtype=float)
        if df.ndim > 1: 
            df = df[:,channel]
            df.flatten()
        x = range(len(df))
        norm = np.sum(df)/1000

        
        peak_coordinates = peak_finder(df/norm)  
        val = valley_finder(df/norm, peak_coordinates[0],peak_coordinates[1])
        valley_coordinates = [val,df[val]/norm]
        
        hwhm = HWHM_right(peak_coordinates,df/norm)


        if type(valley_coordinates[0]) is list:
            peak_to_valley_distance = None
            peak_to_valley_ratio = None
        else:
            peak_to_valley_distance = peak_coordinates[0]-valley_coordinates[0]
            peak_to_valley_ratio = peak_coordinates[1]/valley_coordinates[1]

        print(file)
        print(hwhm)
        print(peak_to_valley_distance)
        print(peak_to_valley_ratio)
        plt.plot(x, df/norm, label = file, alpha = 0.5, )


        f.write("File name: " + file +"\n")
        f.write("Right HWHM: %s lsb\n"%hwhm)
        f.write("Peak to valley distance: %s lsb\n"%peak_to_valley_distance)
        f.write("Peak to valley ratio: %s\n\n" %peak_to_valley_ratio)
    


plt.title("CH %s" %channel)
plt.yscale(LOG)
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.grid()
plt.savefig(os.path.join(FOLDER_PATH,"plot"))
plt.show() 

f.close()
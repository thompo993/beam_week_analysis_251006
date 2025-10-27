import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import personalpaths

FOLDER_PATH = personalpaths.SCAN_PATH#r"path_goes_here" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
VISUALISATION = "PE" #options: "PE", "channel", "both"

xlim = [0, 3000]
ylim = [0, 200]
size = [14,10]

threshold = 250         
p_height = 200        
p_to_v_diff = 0.10      
window = 250    
SHOW_GAUSS = False

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def smooth_data(y, window=70, poly=4):
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)


def findpeak(y):
    x = np.arange(len(y))

    y0= y[500:2500] #take only a slice of data for the gaussian fit  
    x0 = x[500:2500]
    popt, _ = curve_fit(gaussian,x0,y0,p0 = [p_height, 1500,1000], maxfev = 10000000)

    guess = int(popt[1])
    if guess < 750:
        guess = 750
    if guess > 2300:
        guess = 2300
    if guess > 1800:
        ran = 600
    else: ran = window
    x = x[guess-ran:guess+ran]
    y = y[guess-ran:guess+ran]
    popt, _ = curve_fit(gaussian,x,y,p0 = [p_height, guess,popt[2]], maxfev = 10000000)
   
    if SHOW_GAUSS:
        plt.plot(x, gaussian(x,*popt))
    plt.scatter(popt[1], gaussian(popt[1],*popt), color = "k")

    return [popt[1], gaussian(popt[1],*popt)]


def findvalley(y, m_peak, p_h):
    y = -y
    y = y+p_h
    x = np.arange(len(y))
   
    valley_guess = (m_peak+threshold)/2

    min_range = int(valley_guess - window)
    if min_range < threshold:
        min_range = threshold
    max_range = int(valley_guess + window)
    y = y[min_range:max_range] #take only a slice of data for the gaussian fit  
    x = x[min_range:max_range]
    popt, _ = curve_fit(gaussian,x,y,p0 = [p_to_v_diff, valley_guess,valley_guess], maxfev =1000000)

    
    if popt[1]> m_peak:
        mu = 0
    else:    
        mu = popt[1]
        plt.scatter(mu, -gaussian(mu,*popt)+p_h, color = "r")
        if SHOW_GAUSS:
            plt.plot(x, -gaussian(x,*popt)+p_h)

    return [mu, -gaussian(mu,*popt)+p_h]

def peak_to_valley(p,v):
    pv_ratio = p[1]/v[1]
    return pv_ratio






def plot_PE(fold):
    plt.figure(figsize=size)
    for i in range(8):
        for file in os.listdir(fold):
            if "%s.csv" %i in file:
                file_path = os.path.join(fold, file)
                df = np.loadtxt(file_path, dtype = float, delimiter=',')
                x = range(len(df))                
                data = smooth_data(df)
                peak = findpeak(data)
                valley = findvalley(data, peak[0], peak[1])
                peaktovalley = peak_to_valley(peak, valley)
                PtV = "{:.2f}".format(peaktovalley)
                plt.plot(x, data, label = "ch%s - PtV: "%i + PtV)
    plt.title(fold[-5:] + " channel comparison", fontsize = 20)
    plt.xlabel("LSB", fontsize = 20)
    plt.ylabel("Counts", fontsize = 20)
    plt.xlim(xlim)
    plt.yscale(LOG)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.legend(fontsize = 10)
    plt.grid()
    if SAVE_PNG:
        plt.savefig(os.path.join(FOLDER_PATH,fold[-5:]+LOG))
    plt.close()

def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df, label = folder[-5:])
    plt.title("CH_%s PE scan"%num, fontsize = 20)
    plt.xlabel("LSB", fontsize = 20)
    plt.ylabel("Counts", fontsize = 20)
    plt.yscale(LOG)
    # plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(fontsize = 10)
    plt.tight_layout()
    plt.grid()
    if SAVE_PNG:
        plt.savefig(os.path.join(FOLDER_PATH,"CH_%s"%num+LOG))
    plt.close() 


if VISUALISATION == "PE" or VISUALISATION == "both":
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        if "PE" in folder:
            plot_PE(folder)

if VISUALISATION == "channel" or VISUALISATION == "both":
    for i in range(8):
        plot_channels(i)

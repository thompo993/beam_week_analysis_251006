import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import personalpaths

PATH = personalpaths.FOM_PATH#r"path_goes_here" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
VISUALISATION = "PE" #options: "PE", "channel", "both"

xlim = [0, 2000]
ylim = [0, 300]
size = [14,10]

threshold = 180         
p_height = 150        
p_to_v_diff = 100      
window = 150    
valley_window = 150
perc = 0.2
hwhm_ave =0.05
SHOW_GAUSS = False

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def smooth_data(y, window=70, poly=3):
    if window % 2 == 0:
        window += 1
    if window > len(y):
        window = len(y) - 1 if len(y) % 2 == 0 else len(y)
    return savgol_filter(y, window_length=window, polyorder=poly)


def findpeak(y):
    x = np.arange(len(y))

    peaks,_ = find_peaks(y, height=40, width=50)
    if len(peaks)==0:       
        guess =1500
    elif len(peaks)>1:    guess = int(peaks[-1])
    elif len(peaks)==1:   guess = int(peaks[0])
    if guess < threshold:
        guess = 450
    if guess > 2300:
        guess = 2300
    if guess > 1800:
        ran = 600
    elif guess < 500:
        ran = 100
    else: ran = window
    if guess + ran -threshold < 200:
        x1 = x[threshold:threshold +400]
        y1 = y[threshold:threshold +400]
        guess = threshold + 500
    else:
        x1 = x[guess-ran:guess+ran]
        y1 = y[guess-ran:guess+ran]
    popt, _ = curve_fit(gaussian,x1,y1,p0 = [p_height, guess,50], maxfev = 10000000)

    guess = int(popt[1])
    sigma = int(popt[2]*0.6)
    x = x[guess - sigma : guess + sigma]
    y = y[guess - sigma : guess + sigma]

    popt, _ = curve_fit(gaussian,x1,y1,p0 = [p_height, guess,popt[2]/100], maxfev = 10000000)

   
    if SHOW_GAUSS:
        plt.plot(x, gaussian(x,*popt))
    plt.scatter(popt[1], gaussian(popt[1],*popt), color = "b")

    return [popt[1], gaussian(popt[1],*popt)]


def findvalley(y, m_peak, p_h):
    y = -y
    y = y+p_h
    x = np.arange(len(y))
   
    valley_guess = (m_peak+threshold)/2
    valley_window = perc *valley_guess

    min_range = int(valley_guess - valley_window)
    max_range = int(valley_guess + valley_window)
    if min_range < threshold:
        min_range = threshold
        max_range = threshold + 300
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
    pv_distance = p[0]-v[0]
    x = [p[0], p[0]]    
    y = [0, p[1]]
    plt.plot(x,y,  label = "peak height", color = "b")
    x = [v[0], v[0]]    
    y = [0, v[1]]
    plt.plot(x,y,   label = "valley height", color ="r")

    return pv_ratio

def shoulder(peak, valley, data):
    x = []
    y = []
    data = data[threshold:int(valley[0])]
    half = peak[1]
    for k in range(len(data)):
        if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
            x.append(k+threshold)
            y.append(data[k])
    plt.scatter(np.average(x), half, color = 'g')
    x = [np.average(x), peak[0]]    
    y = [half, half]
    plt.plot(x,y,   label = "valley width",color = 'g')
    return [np.average(x), np.average(y)]

def HWHM_right(coordinates, data):
    x = []
    y = []
    half = coordinates[1]/2
    for k in range(int(coordinates[0]), len(data)):
        if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
            x.append(k)
            y.append(data[k])
    plt.scatter(np.average(x), half, color = 'orange')
    x = [coordinates[0], np.average(x)]    
    y = [half,half]
    plt.plot(x,y,   label = "rHWHM", color = 'orange')
    return np.average(x)-coordinates[0]


plt.figure(figsize = [14,10])

df = np.loadtxt(PATH, dtype = float, delimiter=',')
x = range(len(df))                
data = smooth_data(df)
peak = findpeak(data)
valley = findvalley(data, peak[0], peak[1])
peaktovalley = peak_to_valley(peak, valley)
distance = shoulder(peak,valley, data)
width = peak[0]-distance[0]

hwhm = HWHM_right(peak,data) 

PtV = "{:.2f}".format(peaktovalley)


plt.plot(x, data, label = "PHS", linestyle = "dotted")
                    

plt.title("Figure of Merit quantities", fontsize = 20)
plt.xlabel("LSB", fontsize = 20)
plt.ylabel("Counts", fontsize = 20)
plt.xlim(xlim)
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)
plt.yscale(LOG)
plt.ylim(ylim)
plt.tight_layout()
plt.legend(fontsize = 20)
plt.grid()
plt.savefig(r"\\isis\shares\DigitalMuons\supermusr_stavetesting_beam_week_251006\japan_conference_plots_251024\figure_of_merit\quantities.png")
plt.show()

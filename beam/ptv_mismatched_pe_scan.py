import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import personalpaths as personalpaths 

FOLDER_PATH = personalpaths.MISMATCHED_PATH#r"folderpath" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
PE = 70
description = "Module C, ID 19757" #add what you want the title to say, if nothing leave a "" 
if description != "":
    description = " - " + description

xlim = [0, 3500]
ylim = [0, 4] #choose appropriate limits, only applies to linear plots
size = [14,10]

SAVE_PATH = os.path.join(FOLDER_PATH,"figures", "ptv_analysis")
os.makedirs(SAVE_PATH, exist_ok=True)


threshold = 300         
p_height = 0.1       
p_to_v_diff = 0.2      
window = 400    
valley_window = 400
perc = 0.2
hwhm_ave =0.05
SHOW_GAUSS = False



# threshold = 180         
# p_height = 0.2       
# p_to_v_diff = 0.3      
# window = 150    
# valley_window = 150
# perc = 0.2
# hwhm_ave =0.05
# SHOW_GAUSS = True

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

    peaks,_ = find_peaks(y, height=0.4, width=50)
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

    popt, _ = curve_fit(gaussian,x1,y1,p0 = [p_height, guess,popt[2]/10], maxfev = 10000000)

   
    if SHOW_GAUSS:
        plt.plot(x, gaussian(x,*popt))
    plt.scatter(popt[1], gaussian(popt[1],*popt), color = "k")

    return [popt[1], gaussian(popt[1],*popt)]


def findvalley(y, m_peak, p_h):
    if m_peak < 3500:
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
    else: return [0,0]


def peak_to_valley(p,v):
    if v != [0,0]:
        pv_ratio = p[1]/v[1]
        x = [p[0], v[0]]    
        y = [p[1], v[1]]
        plt.plot(x,y, color = "k", alpha = 0.25)
        return pv_ratio
    else: return 0

def shoulder(peak, valley, data):
    if valley != [0,0]:
        x = []
        y = []
        data = data[threshold:int(valley[0])]
        half = peak[1]
        for k in range(len(data)):
            if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
                x.append(k+threshold)
                y.append(data[k])
        plt.scatter(np.average(x), np.average(y), color = 'g')

        return [np.average(x), np.average(y)]
    else: return [0,0]

def HWHM_right(coordinates, data):
    x = []
    y = []
    half = coordinates[1]/2
    for k in range(int(coordinates[0]), len(data)):
        if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
            x.append(k)
            y.append(data[k])
    # plt.scatter(np.average(x), np.average(y), color = 'g')
    return np.average(x)-coordinates[0]


f = open(os.path.join(SAVE_PATH,"mis_log.txt"), "w")



def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH): #looks into all folders 
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file and "amplitude" in file:
                file_path = os.path.join(folder, file)
                df = np.loadtxt(file_path, dtype = float,delimiter=',')
                norm = np.sum(np.array(df))/1000
                df = smooth_data(df, 8)
                data = df/norm
                x = range(len(data))
                PtV = "0"                    
                try:
                    peak = findpeak(data)
                    valley = findvalley(data, peak[0], peak[1])
                    peaktovalley = peak_to_valley(peak, valley)
                    distance = shoulder(peak,valley, data)
                    width = peak[0]-distance[0]
                    hwhm = HWHM_right(peak,data) 

                    PtV = "{:.2f}".format(peaktovalley)                    
                    f.write("\n"+  folder[-3:]+",%s,%s,%s,%s"%(i,peaktovalley,width,hwhm))
                except: print("no")
                plt.plot(x, data, label = "Left \u0394LSB/PE = " + folder[-3:] + " Ptv = "+PtV, linewidth = 0.5)
    plt.title("Mismatched scan, Right \u0394LSB/PE = %s - Channel %s"%(PE,num) + description, fontsize =20)
    plt.yscale(LOG)
    if LOG == 'linear':    
        plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel("LSB", fontsize =20 )
    plt.ylabel("Counts [Area Normalised]", fontsize =20)
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)
    leg = plt.legend(fontsize =10)

    for legobj in leg.legend_handles: #increase size of lines in legend to show colors better
        legobj.set_linewidth(4.0)
    
    plt.tight_layout()
    plt.grid()
    if SAVE_PNG:#if true save, if false show
        plt.savefig(os.path.join(SAVE_PATH,"CH_%s"%num+LOG))
        plt.close()
    else: plt.show() 
    print("Mismatched scan, Left \u0394LSB/PE = %s - Channel %s"%(PE,num) + description)



for i in range(8): #do this for all channels 
    plot_channels(i)

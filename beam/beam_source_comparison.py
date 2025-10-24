import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import personalpaths
from matplotlib.colors import TABLEAU_COLORS, same_color

FOLDER_PATH = personalpaths.FOLDER_PATH_BEAM_SOURCE_COMPARISON # if not using personalpaths file, use r"filepath"
LOG = 'linear' #options: 'log', 'linear'
channel = 5
PE = 25
MODULE = "BB"

xlim = [0, 2500]
ylim = [0, 4]
size = [14,10]

#parameters for peak finding 
threshold = 250         #lower limit in xfor peak finding
p_height = 0.76         #lower limit in y for peak finding 
p_to_v_diff = 0.10      #mimimum difference between peak and valley for valley finding
hwhm_ave = 0.01         #percentage of max_height used to take a slice to calculate the HWHM
ran = 200               #half-width of slice taken to fit gaussian peak

SHOW_GAUSS = False       #show gaussian fit on plot 

f = open(os.path.join(FOLDER_PATH, 'log.txt'), 'w')

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
    x = np.arange(len(y))
    if SHOW_GAUSS:
        ax.plot(y)
    #find guess for gaussian
    peaks, _ = find_peaks(y, height = p_height)
    cut = 0
    for i in range(len(peaks)): #remove peaks that are too far apart
        if peaks[i]-peaks[i-1] > ran:
            cut = i
    peaks = peaks[cut:]
    pk = [i for i in peaks if i>threshold]
    # peak_height_max = np.max(y[pk])
    # pk = [i for i in peaks if y[i]>0.80*peak_height_max]
    if len(pk) == 0:
        print("Unable to find peaks. Change value of p_height")

    #fit around the peaks that were found with a gaussian of high sigma
    peak_guess = np.average(pk)
    min_range = int(peak_guess - ran)
    if min_range < threshold:
        min_range = threshold
    max_range = int(peak_guess + ran)
    y = y[min_range:max_range] #take only a slice of data for the gaussian fit  
    x = x[min_range:max_range]
    popt, _ = curve_fit(gaussian,x,y,p0 = [p_height, peak_guess,peak_guess/2], maxfev = 10000000)
   
    if SHOW_GAUSS:
        ax.plot(x, gaussian(x,*popt))
    ax.scatter(popt[1], gaussian(popt[1],*popt), color = "k")

    return [popt[1], gaussian(popt[1],*popt)]

def valley_finder(data, m_peak, p_h):
    y = smooth_data(-data)
    y = y+p_h
    x = np.arange(len(y))
   
    valley_guess = (m_peak+threshold)/2

    min_range = int(valley_guess - ran)
    if min_range < threshold:
        min_range = threshold
    max_range = int(valley_guess + ran)
    y = y[min_range:max_range] #take only a slice of data for the gaussian fit  
    x = x[min_range:max_range]
    popt, _ = curve_fit(gaussian,x,y,p0 = [p_to_v_diff, valley_guess,valley_guess], maxfev =1000000)

    
    if popt[1]> m_peak:
        mu = 0
    else:    
        mu = popt[1]
        ax.scatter(mu, -gaussian(mu,*popt)+p_h, color = "r")
        if SHOW_GAUSS:
            ax.plot(x, -gaussian(x,*popt)+p_h)

    return [mu, -gaussian(mu,*popt)+p_h]

def HWHM_right(coordinates, data):
    data=smooth_data(data)
    x = []
    y = []
    half = coordinates[1]/2
    for k in range(int(coordinates[0]), len(data)):
        if data[k] < half*(1+hwhm_ave) and data[k] > half*(1-hwhm_ave):
            x.append(k)
            y.append(data[k])
    ax.scatter(np.average(x), np.average(y), color = 'g')
    return np.average(x)-coordinates[0]
    

f.write("CHANNEL %s\n"%channel)
f.write("Module " + MODULE +"\n") 
f.write("PE %s\n\n"%PE)

f.write("Analysis parameters:\n")
f.write("threshold: %s\n" %threshold)
f.write("p_height: %s\n" %p_height)
f.write("p_to_v_diff: %s\n" %p_to_v_diff)
f.write("hwhm_ave: %s\n" %hwhm_ave)
f.write("ran: %s\n\n" %ran)


fig, ax = plt.subplots(figsize=size)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

col = 0

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

        rebin = []
        tmp = [df[i] for i in range(len(df))]
        
        for k in range(16):
            tmp.append(0)

        for i in range(len(df)):
            bin = tmp[i-8:i+8]
            rebin.append(np.average(bin)/norm)
        ax.plot(x, rebin, label = file[:-4], alpha = 0.5, color = colors[col])


        
        peak_coordinates = peak_finder(df/norm)  
        valley_coordinates = valley_finder(df/norm, peak_coordinates[0],peak_coordinates[1])
        
        hwhm = HWHM_right(peak_coordinates,df/norm)

        if valley_coordinates[0] != 0:
            peak_to_valley_distance = peak_coordinates[0]-valley_coordinates[0]
            peak_to_valley_ratio = peak_coordinates[1]/valley_coordinates[1]
        else: 
            peak_to_valley_distance = None
            peak_to_valley_ratio = None


        # print(file)
        # print(peak_coordinates)
        # print(valley_coordinates)

        f.write("File name: " + file +"\n")
        f.write("Right HWHM: %s lsb\n"%hwhm)
        f.write("Peak to valley distance: %s lsb\n"%peak_to_valley_distance)
        f.write("Peak to valley ratio: %s\n\n" %peak_to_valley_ratio)

        infotext = file[:-4] +"°\n"
        infotext += "Right-hand HWHM: {:.0f} lsb\n".format(hwhm) 
        if peak_to_valley_distance != None:
            infotext += "P_to_V distance: {:.0f} lsb\n".format(peak_to_valley_distance)
            infotext += "P_to_V ratio: {:.2f}".format(peak_to_valley_ratio)
        else:
            infotext += "P_to_V distance: N/A\n"
            infotext += "P_to_V ratio: N/A"

        props = dict(boxstyle='round', facecolor=colors[col], alpha=0.30)

        ax.text(0.98, 0.95-(col/8), infotext, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        col += 1

ax.set_xlabel("LSB")
secx = ax.secondary_xaxis('top')
secx.set_xticklabels(["{:.0f}".format(x/PE) for x in ax.get_xticks()])
secx.set_xlabel("Photons")
ax.set_ylabel("Counts (normalised to area 1000)")
ax.grid()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("Channel %s - Module "%channel + MODULE)
ax.set_yscale(LOG)


# plt.legend()
if SHOW_GAUSS:
    filename = "plot_gauss" 
else: filename = "plot"
plt.savefig(os.path.join(FOLDER_PATH,filename))
plt.show() 

f.close()
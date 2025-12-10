import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import personalpaths
import json
plt.rcParams.update({'font.size': 20})


FOLDER_PATH = personalpaths.SCAN_PATH#r"path_goes_here" #folderpath goes here, ideally nothing else in the folder

SAVE_PATH = os.path.join(FOLDER_PATH, "figures")
os.makedirs(SAVE_PATH,exist_ok=True)
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
VISUALISATION = "both" #options: "PE", "channel", "both"
rebin_factor = 4

props = dict(boxstyle='round', facecolor="white", alpha=0.60)


def rebin(x, factor:int):
    rebin = []
    zero = np.zeros(factor)
    tmp = np.append(zero, x)
    tmp = np.append(tmp,zero)
    for i in range(len(x)):
        bin = np.average(tmp[i-factor: i+factor])
        rebin.append(bin)
    return rebin


def PE_line(PE, slope, intercept):
    return PE*slope +intercept

for files in os.listdir(FOLDER_PATH):
    if "json" in files:
        with open(os.path.join(FOLDER_PATH,files), 'r') as file:
            sipm_info = json.load(file)
        

Lindex = []
Lslope = []
Lintercept = []
Rindex = []
Rslope = []
Rintercept = []

for i in range(16):
    try:
        k = str(i)
        if "LEFT" in sipm_info["sipm"][k]["position"]:
            Lindex.append(sipm_info["sipm"][k]["P"])
            Lslope.append(sipm_info["sipm"][k]["hv_calibration"]["m"])
            Lintercept.append(sipm_info["sipm"][k]["hv_calibration"]["q"])
        if "RIGHT" in sipm_info["sipm"][k]["position"]:
            Rindex.append(sipm_info["sipm"][k]["P"])
            Rslope.append(sipm_info["sipm"][k]["hv_calibration"]["m"])
            Rintercept.append(sipm_info["sipm"][k]["hv_calibration"]["q"])
    except: print("missing %s" %i )

Lindex = [int(x) for x in Lindex]
Rindex = [int(x) for x in Rindex]

Lslope = [float(x) for x in Lslope]
Rslope = [float(x) for x in Rslope]
Lintercept = [float(x) for x in Lintercept]
Rintercept = [float(x) for x in Rintercept]


print(Lslope)
print(Rslope)

xlim = [0, 4095]
ylim = [0, 300]
size = [14,10]

def plot_PE(fold):
    fig =plt.figure(figsize=size)
    ax = fig.add_subplot()
    text = ""
    for i in range(8):
        for file in os.listdir(fold):
            if "%s.csv" %i in file and "amplitude" in file:
                try:
                    PE = int(fold[-3:])
                    file_path = os.path.join(fold, file)
                    df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                    x = range(len(df))
                    df = rebin(df,rebin_factor)
                    ax.plot(x, df, label = "ch%s"%i)
                    
                    Linterceptch = Lintercept[i]
                    Rinterceptch = Rintercept[i]
                    Lslopech = Lslope[i]
                    Rslopech = Rslope[i]

                    Lbias = PE_line(PE,Lslopech, Linterceptch)
                    Rbias = PE_line(PE,Rslopech,Rinterceptch)


                    text += "ch%s"%i + ": LBIAS = {:.2f}, LVBR = {:.2f}, RBIAS = {:.2f}, RVBR = {:.2f}\n".format(Lbias-2.5, Linterceptch,Rbias-2.5,Rinterceptch)
                except: 
                    print("Missing channel information for channel %s"%i)
    plt.suptitle("Channel comparison, \u0394LSB/PE = " +fold[-3:], fontsize = 20)
    ax.set_xlabel("LSB")
    ax.set_ylabel("Counts") 
    ax.set_xlim(xlim)
    ax.set_yscale(LOG)
    ax.set_ylim(ylim)
    plt.tight_layout()
    ax.legend(fontsize =15)
    plt.grid()


    ax.text(0.98, 0.95, text[:-1], transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='right',bbox = props)

    if SAVE_PNG:
        plt.savefig(os.path.join(SAVE_PATH,fold[-5:]+LOG))

    plt.close()

def plot_channels(num):
    fig =plt.figure(figsize=size)
    ax = fig.add_subplot()
    text =""
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file and "amplitude" in file:
                try:
                    PE = int(folder[-3:])

                    file_path = os.path.join(folder, file)
                    df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                    x = range(len(df))
                    df = rebin(df,rebin_factor)

                    ax.plot(x, df, label = "\u0394LSB/PE = " + folder[-3:])


                    Linterceptch = Lintercept[i]
                    Rinterceptch = Rintercept[i]
                    Lslopech = Lslope[i]
                    Rslopech = Rslope[i]

                    Lbias = PE_line(PE,Lslopech, Linterceptch)
                    Rbias = PE_line(PE,Rslopech,Linterceptch)


                    text += "ch%s"%i + ": LBIAS = {:.2f}, LVBR = {:.2f}, RBIAS = {:.2f}, RVBR = {:.2f}\n".format(Lbias, Linterceptch,Rbias,Rinterceptch)


                except:
                    print("missing information")
    plt.suptitle("CH_%s \u0394LSB/PE scan"%num, fontsize = 20)
    ax.set_xlabel("LSB")
    ax.set_ylabel("Counts") 
    ax.set_xlim(xlim)
    ax.set_yscale(LOG)
    ax.set_ylim(ylim)
    plt.tight_layout()
    ax.legend(fontsize = 15)
    plt.grid()

    ax.text(0.02, 0.95, text[:-1], transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='left',bbox = props)

    if SAVE_PNG:
        plt.savefig(os.path.join(SAVE_PATH,"CH_%s"%num+LOG))
    plt.close() 


if VISUALISATION == "PE" or VISUALISATION == "both":
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        if "PE" in folder:
            plot_PE(folder)

if VISUALISATION == "channel" or VISUALISATION == "both":
    for i in range(8):
        plot_channels(i)

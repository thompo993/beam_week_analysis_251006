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
                PE = int(fold[-3:])
                file_path = os.path.join(fold, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                ax.plot(x, df, label = "ch%s"%i)

                text += "ch%s"%i + ": LBIAS = , LVBR = , RBIAS = , RVBR = \n"
    plt.suptitle("Channel comparison, \u0394LSB/PE = " +fold[-3:], fontsize = 20)
    ax.set_xlabel("LSB")
    ax.set_ylabel("Counts") 
    ax.set_xlim(xlim)
    ax.set_yscale(LOG)
    ax.set_ylim(ylim)
    plt.tight_layout()
    ax.legend()
    plt.grid()

    ax.text(0.98, 0.95, text[:-1], transform=ax.transAxes, fontsize=15, verticalalignment='top', horizontalalignment='right')

    if SAVE_PNG:
        plt.savefig(os.path.join(SAVE_PATH,fold[-5:]+LOG))

    plt.close()

def plot_channels(num):
    plt.figure(figsize=size)
    for folder in os.walk(FOLDER_PATH):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%num in file and "amplitude" in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df, label = "\u0394LSB/PE = " + folder[-3:])
    plt.title("CH_%s \u0394LSB/PE scan"%num, fontsize = 20)
    plt.xlabel("LSB")
    plt.ylabel("Counts")
    plt.yscale(LOG)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()
    plt.grid()
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

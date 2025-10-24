import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import personalpaths

FOLDER_PATH = personalpaths.SCAN_PATH#r"path_goes_here" #folderpath goes here, ideally nothing else in the folder
SAVE_PNG = True
LOG = 'linear' #options: 'log', 'linear'
VISUALISATION = "PE" #options: "PE", "channel", "both"

xlim = [0, 3000]
ylim = [0, 200]
size = [14,10]

def plot_PE(fold):
    plt.figure(figsize=size)
    for i in range(8):
        for file in os.listdir(fold):
            if "%s.csv" %i in file:
                file_path = os.path.join(fold, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df, label = "ch%s"%i)
    plt.title(fold[-5:] + " channel comparison", fontsize = 20)
    plt.xlabel("LSB")
    plt.ylabel("Counts")
    plt.xlim(xlim)
    plt.yscale(LOG)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.legend()
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
    plt.xlabel("LSB")
    plt.ylabel("Counts")
    plt.yscale(LOG)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
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

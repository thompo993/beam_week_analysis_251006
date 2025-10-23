import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys

FOLDER_PATH_source = r"pathhere"#folderpath goes here for source data 
FOLDER_PATH_beam = r"pathhere"#folderpath goes here for beam data
FOLDER_PATH_save = r""#folderpath where to save photos

if not os.path.isdir(FOLDER_PATH_beam):
    beam = False
else: beam = True

if not os.path.isdir(FOLDER_PATH_save):
    FOLDER_PATH_save = FOLDER_PATH_source

if not os.path.isdir(FOLDER_PATH_source):
    print("error: source folder does not exist")
    sys.exit(0)

SAVE_PNG = True
channel = [1,5] #choose channel you want to visualize, in a list 
LOG = 'log' #options: 'log', 'linear'

xlim = [None, 3000]
ylim = [0, None]
size = [10,8]
norm = 20


plt.figure(figsize=size)
for i in channel:
    for folder in os.walk(FOLDER_PATH_beam):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%i in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df/norm, label = folder[-5:]+ " ch %s - beam"%i)
    for folder in os.walk(FOLDER_PATH_source):
        folder = folder[0]
        for file in os.listdir(folder):
            if "%s.csv"%i in file:
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
                x = range(len(df))
                plt.plot(x, df, label = folder[-5:]+ " ch %s - Sr90"%i)
plt.title("All PE on chosen channels")
plt.yscale(LOG)
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.grid()
if SAVE_PNG:
    plt.savefig(os.path.join(FOLDER_PATH_save,"all_"+LOG+str(channel)))
plt.show() 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import sys

FOLDER_PATH = r"C:\Users\xph93786\Desktop\first_plots"#folderpath goes here


SAVE_PNG = False
LOG = 'linear' #options: 'log', 'linear'
channel = [1]

xlim = [None, 3000]
ylim = [0, None]
size = [10,8]


plt.figure(figsize=size)
for file in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file)
    if "amplitude" in file:
        df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True)
    else:
        df = pd.read_csv(file_path, on_bad_lines='skip', na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"], skipinitialspace=True,usecols= channel)
    x = range(len(df))
    norm = np.sum(df)
    plt.plot(x, df/norm, label = file)

plt.title("CH %s" %channel)
plt.yscale(LOG)
plt.xlim(xlim)
plt.ylim(ylim)
plt.legend()
plt.grid()

plt.show() 


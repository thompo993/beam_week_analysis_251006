import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import personalpaths as pp
import os

file_list = []
for file in os.listdir(pp.folder_phs):
    if file[-3:]=="csv":
        file_list.append(file)

for file in file_list:
    data= np.loadtxt(os.path.join(pp.folder_phs, file), dtype=float, delimiter=",",skiprows=1)
    y,bin_edge = np.histogram(data, bins=70, range =[150,500])
    plt.hist(data, bins=100, range = [0,500])
    plt.savefig(os.path.join(pp.folder_phs, file[:-4]))
    plt.close()
    print((np.argmax(y)+30)*5)
    
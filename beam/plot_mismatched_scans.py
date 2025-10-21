import matplotlib.pyplot as plt
import numpy as np
import os

#data needs to be in a csv with the last 4 digits are pe1 and pe2

pe1 = 20
pe2 = 35

plt.figure(figsize = [12,10])
title  = "PE%s-PE%s" %(pe1,pe2)

for file in os.listdir(r".\data"):
    if file.endswith(".csv"):
        if (str(pe1)+str(pe1) in file) or (str(pe1)+str(pe2) in file) or (str(pe2)+str(pe1) in file) or (str(pe2)+str(pe2) in file):
            data = np.loadtxt(os.path.join(".\data",file), dtype = float, delimiter  =",")
            plt.plot(data, label = file[-8:-6] + " " + file[-6:-4])
        
plt.title(title+ " comparison - Channel 0")
plt.grid()
plt.legend()
plt.yscale('log')
plt.savefig(title)
plt.show()    

#runtime for the mismatched scans is 45 minutes, unknown for the matched scan 
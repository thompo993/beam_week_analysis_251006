import matplotlib.pyplot as plt
import numpy as np
import os
import personalpaths as personalpaths

#data needs to be in a csv with the last 4 digits are pe1 and pe2



PATH = personalpaths.MISMATCHED_PATH
rebinning = False

def rebin(x):
    rebin = []
    zero = np.zeros(8)
    tmp = np.append(zero, x)
    tmp = np.append(tmp,zero)
    for i in range(len(x)):
        bin = np.average(tmp[i-8: i+8])
        rebin.append(bin)
    return rebin


def plot(a,b):
    pe1 = a    
    pe2 = b
    plt.figure(figsize = [14,10])
    title  = "PE%s-PE%s" %(pe1,pe2)

    for file in os.listdir(os.path.join(PATH,"data")):
        if file.endswith(".csv"):
            if (str(pe1)+str(pe1) in file) or (str(pe1)+str(pe2) in file) or (str(pe2)+str(pe1) in file) or (str(pe2)+str(pe2) in file):
                data = np.loadtxt(os.path.join(PATH,"data",file), dtype = float, delimiter  =",")
                data = data*1000/np.sum(data)
                if rebinning:

                    data = rebin(data)

                plt.plot(data, label = file[-8:-6] + "  " + file[-6:-4])
            
    plt.title(title+ " comparison - Channel 0", fontsize = 20)
    plt.grid()
    # plt.ylim(0,4)
    plt.xlim(0,2500)
    plt.xlabel("LSB", fontsize = 20)
    plt.ylabel("Counts", fontsize = 20)
    plt.legend()

    plt.yscale('log')
    plt.tight_layout()

    plt.savefig(os.path.join(PATH, title))
    plt.close()    

pe = ["05",10,15,20,25,28,30,32,35,40,50]

for i in pe:
    for k in pe:
        if int(i) > int(k):
            plot(i,k)


#runtime for the mismatched scans is 45 minutes, unknown for the matched scan 
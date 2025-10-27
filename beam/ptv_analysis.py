import personalpaths as personalpaths
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


PATH = personalpaths.PTVLOG_PATH
data = np.loadtxt(PATH, dtype = float, delimiter =",", skiprows =1)

plt.figure(figsize = [14,10])
print(data)

pe = np.unique(data[:,0])

for i in range(len(pe)):
    x = []
    y = []
    k = pe[i]
    for j in range(np.shape(data)[0]):
        if data[j,0] == k:
            x.append(data[j,2])
            y.append(data[j,3])
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x,x/y, label = k)
plt.ylabel("ptv/width")
plt.xlabel("ptv")
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()
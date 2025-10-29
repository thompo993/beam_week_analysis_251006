import personalpaths as personalpaths
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


PATH = personalpaths.PTVLOG_PATH
data = np.loadtxt(os.path.join(PATH, "log.txt"), dtype = float, delimiter =",", skiprows =1)


pe = data[:,0]
channel = data[:,1]
ptv = data[:,2]
width = data[:,3]
hwhm = data[:,4]



plt.figure(figsize = [14,10])
plt.scatter(pe,ptv/width)
plt.title("Figure of Merit")
plt.ylabel("ptv/width")
plt.xlabel("PE")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PATH, "fig4"))
plt.show()

# plt.figure(figsize = [14,10])
# plt.scatter(pe,ptv)
# plt.title("PtV vs PE")
# plt.ylabel("PtV")
# plt.xlabel("PE")
# plt.grid()
# plt.legend()
# plt.savefig(os.path.join(PATH, "fig2"))
# plt.show()


# plt.figure(figsize = [14,10])
# plt.scatter(pe,ptv*(width-hwhm))
# plt.title("Figure of Merit")
# plt.ylabel("ptv/width")
# plt.xlabel("PE")
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(PATH, "fig3"))
# plt.show()

# plt.figure(figsize = [14,10])
# plt.scatter(pe,ptv*width/hwhm)
# plt.title("Figure of Merit")
# plt.ylabel("ptv/width")
# plt.xlabel("PE")
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(PATH, "fig3"))
# plt.show()

plt.figure(figsize = [14,10])

fom = []
pe_plot = []
pe_unique = np.unique(pe)
sigma =[]

for i in range(len(pe_unique)):
    x = []
    pe_plot.append(pe_unique[i])
    for j in range(len(pe)):
        if pe[j] == pe_unique[i]:
            x.append(ptv[j]*width[j]/hwhm[j])
    x = np.array(x)
    fom.append(np.average(x))
    sigma.append(np.std(x))
    

plt.errorbar(pe_plot,fom, yerr = sigma, elinewidth = 1, linewidth = 0, marker = 'o', capsize=5)
plt.title("Figure of Merit")
plt.ylabel("ptv*width/hwhm")
plt.xlabel("PE")
plt.grid()
# plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PATH, "fig2"))
plt.show()


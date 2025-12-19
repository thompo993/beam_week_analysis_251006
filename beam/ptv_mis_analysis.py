import personalpaths as personalpaths
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


PATH = personalpaths.PTVLOG_PATH
data = np.loadtxt(os.path.join(PATH, "mis_log.txt"), dtype = float, delimiter =",", skiprows =1)

print(data)
data = data[~np.isnan(data).any(axis=1)]
print(data)

pe = data[:,0]
channel = data[:,1]
ptv = data[:,2]
width = data[:,3]
hwhm = data[:,4]



plt.figure(figsize = [14,10])
plt.scatter(pe,ptv/width)
plt.title("Figure of Merit")
plt.ylabel("ptv/width")
plt.xlabel("\u0394LSB/PE")
plt.xlim(0, 150)
plt.ylim(0, 0.008)


plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(PATH, "fig4_mis"))
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
    
print(fom)

plt.errorbar(pe_plot,fom, yerr = sigma, elinewidth = 1, linewidth = 0, marker = 'o', capsize=5)
plt.title("Figure of Merit, mismatched (Left 30 \u0394LSB/PE)")
plt.ylabel("ptv*width/hwhm")
plt.xlabel("\u0394LSB/PE")
plt.xlim(0, 150)
plt.grid()
# plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PATH, "fig2_mis"))
plt.show()


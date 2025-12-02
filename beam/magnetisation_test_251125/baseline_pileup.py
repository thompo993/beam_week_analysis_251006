import matplotlib.pyplot as plt
import numpy as np
import personalpaths as pp
import os
from scipy import special
from scipy.optimize import curve_fit


file_list, order_list = pp.btest_dist(pp.folder_histos)
sequence = ["56 V - 0 T","56 V - ramp up","56 V - 1.3 T","54 V - 1.3 T","54 V - ramp down","54 V - 0 T"]


def skewnorma(x,a,alpha,sigma,x0):
    norm = 1/(sigma*np.sqrt(2*np.pi))
    exp_factor=-0.5*((x-x0)/sigma)**2
    erf_factor= alpha*((x-x0)/(sigma*np.sqrt(2)))
    return a*norm*np.exp(exp_factor)*(1+special.erf(erf_factor))


for file in file_list:
    data = np.loadtxt(os.path.join(pp.folder_histos,file), dtype = float, delimiter=",",skiprows=1)
    x, bins = np.histogram(data, bins =300, range = [0,2000])
    y = x/sum(x)
    bin_cent = (bins[:-1] + bins[1:])/2
    # label = sequence[int(file[0])]
    plt.plot(bin_cent, y)
    plt.xlabel("LSB")
    plt.title("Baseline comparison")
    plt.legend()
    plt.grid()  
plt.show()

# mu = []
# sigma = []
# mu_err = []
# sigma_err =[]
# alpha=[]
# alpha_err=[]


# for file in file_list:
#     data = np.loadtxt(os.path.join(pp.folder_histos,file), dtype = float, delimiter=",",skiprows=1)
#     x = data[:,1]
#     y = data[:,0]
#     xfit = np.linspace(x[0], x[-1],1000)

#     popt,pcov = curve_fit(skewnorma,x,y,p0= [1e8, -0.1, 1, 90], sigma=np.sqrt(y))
#     print(popt)
#     plt.scatter(x,y, label = "exp")
#     plt.plot(xfit, skewnorma(xfit, *popt), label = "fit")
#     plt.xlabel("LSB")
#     plt.title(file[2:-4])
#     plt.legend()
#     plt.grid()
#     plt.savefig(os.path.join(pp.folder_histos,file[:-4]+".png"))
#     plt.show()
    
#     mu.append(popt[-1])
#     mu_err.append(np.sqrt(pcov[-1,-1]))
#     sigma.append(popt[-2])
#     sigma_err.append(np.sqrt(pcov[-2,-2]))
#     alpha.append(popt[1])
#     alpha_err.append(np.sqrt(pcov[1,1]))


# plt.figure(figsize=[9,6])
# plt.errorbar(order_list, mu,yerr = mu_err, elinewidth=1, fmt='o', capsize=5)
# plt.title("Mean baseline")
# plt.ylabel("LSB")
# plt.xlabel("Sequence")
# plt.xticks(order_list, sequence)
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(pp.folder_histos,"mean.png"))
# plt.show()

# plt.figure(figsize=[9,6])
# plt.errorbar(order_list, sigma, yerr = sigma_err, elinewidth=1, fmt='o', capsize=5)
# plt.title("Sigma baseline")
# plt.ylabel("LSB")
# plt.xlabel("Sequence")
# plt.xticks(order_list, sequence)
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(pp.folder_histos,"sigma.png"))
# plt.show()

# plt.figure(figsize=[9,6])
# plt.errorbar(order_list, alpha,yerr = alpha_err, elinewidth=1, fmt='o', capsize=5)
# plt.title("Skew")
# plt.ylabel("Alpha")
# plt.xlabel("Sequence")
# plt.xticks(order_list, sequence)
# plt.tight_layout()
# plt.grid()
# plt.savefig(os.path.join(pp.folder_histos,"alpha.png"))
# plt.show()

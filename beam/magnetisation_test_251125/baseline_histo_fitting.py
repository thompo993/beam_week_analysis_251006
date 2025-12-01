from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt()

ae, loce, scalee = stats.skewnorm.fit(data)
plt.figure()

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.skewnorm.pdf(x,ae, loce, scalee)#.rvs(100)
plt.plot(x, p, 'k', linewidth=2)
plt.show()

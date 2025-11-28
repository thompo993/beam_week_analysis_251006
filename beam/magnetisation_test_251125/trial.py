import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(1,1,1000)
y = np.random.normal(1,1,1000)

fig1 = plt.figure()
fig2 = plt.figure()
fig1.suptitle("hello")
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)

ax1.scatter(x,y)
ax2.set_title("hello")
ax2.scatter(x*100,y*100)

fig1.savefig("hi")

fig1 = plt.figure()

ax1 = fig1.add_subplot(111)

# ax1.scatter(x*1040,y*100, label = "gf")
ax1.plot(np.arange(len(y)),y)
ax1.scatter(np.arange(len(y)),y)
ax1.legend()
ax1.grid()
fig1.savefig("hi2")

plt.show()

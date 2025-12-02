import numpy as np 
import matplotlib.pyplot as plt
import os
import personalpaths as pp
plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams.update({'font.size': 6})


file_list = pp.btest_csvfiles(pp.folder_btest)
SAVE_PATH = os.path.join(pp.folder_btest,'save_figures')
os.makedirs(SAVE_PATH, exist_ok=True)



for file in file_list:
    filepath = os.path.join(pp.folder_btest,file)
    data = np.loadtxt(filepath, dtype=float, delimiter=',')

    fig, axs = plt.subplots(4,2,sharex=True, sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    fig.suptitle(file[:-4], fontsize= 15)
    for i in range(8):
        k = int(i/2)
        j = i%2
        y = data[:-5,i]/sum(data[:-5,i])
        axs[k, j].plot(y)
        axs[k,j].set_title('Channel %s' %i, fontsize=10)
    for ax in axs.flat:
        ax.set(xlabel="LSB", ylabel="Normalised counts")
        # ax.set(ylim=[-10, 100])
    plt.savefig(os.path.join(SAVE_PATH, file[:-4] + ".png"))
    # plt.show()
    plt.close()



fig, axs = plt.subplots(4,2,sharex=True, sharey=True)
fig.set_figheight(8)
fig.set_figwidth(10)
fig.suptitle("Overlaid", fontsize= 15)

for file in file_list:
    filepath = os.path.join(pp.folder_btest,file)
    data = np.loadtxt(filepath, dtype=float, delimiter=',')
    for i in range(8):
        k = int(i/2)
        j = i%2
        y = data[:-5,i]/sum(data[:-5,i])
        axs[k, j].plot(y, linewidth = 0.5, alpha = 0.8)
        axs[k,j].set_title('Channel %s' %i, fontsize=10)
for ax in axs.flat:
    ax.set(xlabel="LSB", ylabel="Normalised counts", yscale = "log")
        # ax.set(ylim=[-10, 100])
plt.savefig(os.path.join(SAVE_PATH, "overlaid.png"))
# plt.show()
plt.close()



fig, axs = plt.subplots(len(file_list),1,sharex=True, sharey=True)
fig.set_figheight(8)
fig.set_figwidth(10)
fig.suptitle("Channel 0", fontsize= 15)

i = 0
for file in file_list:
    filepath = os.path.join(pp.folder_btest,file)
    data = np.loadtxt(filepath, dtype=float, delimiter=',')
    y = data[:-5,0]/sum(data[:-5,0])
    axs[i].plot(y)
    axs[i].set_title(file[:-4], fontsize=10)
    for ax in axs.flat:
        ax.set(xlabel="LSB", ylabel="Normalised counts")
        # ax.set(ylim=[-10, 100])
    # plt.savefig(os.path.join(SAVE_PATH, x_label + y_label + "_scatter.png"))
    i += 1
plt.savefig(os.path.join(SAVE_PATH, "channel0.png"))
plt.show()
plt.close()


V = [54,56]
for k in V:
    lim_list = [j for j in file_list if str(k) in j]
    fig, axs = plt.subplots(len(lim_list),1,sharex=True, sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(10)
    fig.suptitle("%s V"%k, fontsize= 15)

    i = 0
    for file in lim_list:
        filepath = os.path.join(pp.folder_btest,file)
        data = np.loadtxt(filepath, dtype=float, delimiter=',')
        y = data[:-5,0]/sum(data[:-5,0])
        axs[i].plot(y)
        axs[i].set_title(file[:-4], fontsize=10)
        for ax in axs.flat:
            ax.set(xlabel="LSB", ylabel="Normalised counts")
            # ax.set(ylim=[-10, 100])
        # plt.savefig(os.path.join(SAVE_PATH, x_label + y_label + "_scatter.png"))
        i += 1
    plt.savefig(os.path.join(SAVE_PATH, "%sV.png"%k))
    plt.show()
    plt.close()

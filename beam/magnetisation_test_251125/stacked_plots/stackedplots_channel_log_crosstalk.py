import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import correlate, correlation_lags
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math
from scipy.stats import linregress
import os
from datetime import datetime
import personalpaths as pp

file_list = pp.btest_wavefiles(pp.folder_btest)
current_date_time = datetime.today().strftime('%y%m%d_%H%M%S')


SAVE_CSV = False
xlim = [None,100]
save = True #options: True, False, "expo", "bins"
show = False
pileup_rejection = 1000
size = [15, 10]
channel_list = range(8)



class SuperMUSRBinaryWave():
    def __init__(self):
        self.f = None
        
    def load_file(self, file_name):
        try:
            self.f = open(file_name, 'rb')
            self.f.seek(0)
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found".format(file_name))
            self.f = None
        except Exception as e:
            raise e
            self.f = None
            
    def close_file(self):
        if self.f is not None:
            self.f.close()
            self.f = None
            
    def get_event(self):
        if self.f is None:
            return None
        try:
            return np.load(self.f)
        except Exception:
            return None





# function to process and update the average of exponential signals
def process_events_incremental(reader, prominence=100, filename = "", save_path = "", wave_min=10):
    waves = 0

    while True:
        event = reader.get_event()
        fig, axs = plt.subplots(8,sharex=True, sharey=True)
        fig.set_figheight(30)
        fig.set_figwidth(40)
        fig.suptitle(filename)
        for channel_index in channel_list:
            y_data = event[channel_index]
            axs[channel_index].plot(y_data[3000:6000])
            axs[channel_index].set_yscale("log")
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename[:-4]+"_%s.png"%(waves)))
        plt.close()
        waves +=1
        print(waves)

        if waves>wave_min:
            break
    return 0

# Usage
if __name__ == "__main__":
    SAVE_PATH = os.path.join(pp.folder_btest,'save_figures',"non_averaged_long_channel","log")

    for file in file_list:
        print(file)
        os.makedirs(SAVE_PATH, exist_ok=True)

        file_name = os.path.join(pp.folder_btest,file)
        reader = SuperMUSRBinaryWave()
        reader.load_file(file_name)

        max_events = 10000000000000000000 # maximum n of events 

        k = process_events_incremental(
            reader,
            prominence=20,
            filename = file, 
            save_path = SAVE_PATH,
            wave_min=10
        )

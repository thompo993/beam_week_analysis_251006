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
ylim = [0, 2200]
save = True #options: True, False, "expo", "bins"
show = False
pileup_rejection = 1000
size = [15, 10]
channel_list = [0]
yticks = np.arange(0,ylim[1],200)



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
def process_events_incremental(reader, max_events=1000, prominence=100, pre_points=20, post_points=50, channel_index = 0, filename = "", save_path = "", wave_min=10):
    waves = 0

    while True:
        event = reader.get_event()

        y_data = event[channel_index]
        peaks, _ = find_peaks(y_data, height =(1800, 2200), prominence=prominence)
        print(peaks)

        for i in range(len(peaks)):
            plt.title(filename)
            plt.ylim(ylim)
            plt.grid()
            plt.plot(y_data[peaks[i]-50:peaks[i]+100])
            plt.savefig(os.path.join(save_path, filename[:-4]+"_%s.png"%waves))
            plt.close()
            waves += 1

        if waves>wave_min:
            break
    return 0

# Usage
if __name__ == "__main__":
    SAVE_PATH = os.path.join(pp.folder_btest,'save_figures',"non_averaged")

    for file in file_list:
        print(file)
        os.makedirs(SAVE_PATH, exist_ok=True)

        file_name = os.path.join(pp.folder_btest,file)
        for channel in channel_list:
            reader = SuperMUSRBinaryWave()
            reader.load_file(file_name)

            max_events = 10000000000000000000 # maximum n of events 

            k = process_events_incremental(
                reader,
                max_events=max_events,
                prominence=20,
                pre_points=20,
                post_points=320,
                channel_index = channel,
                filename = file, 
                save_path = SAVE_PATH,
                wave_min=10
            )

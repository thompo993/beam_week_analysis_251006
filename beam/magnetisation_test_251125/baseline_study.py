import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from datetime import datetime
import sys
import personalpaths as pp

file_list = pp.btest_wavefiles(pp.folder_btest)
current_date_time = datetime.today().strftime('%y%m%d_%H%M%S')


SAVE_CSV = False
xlim = [None,100]
ylim = [0, 2000]
save = True #options: True, False, "expo", "bins"
show = False
pileup_rejection = 1000
size = [15, 10]
channel_list = [0]
yticks = np.arange(0,ylim[1],200)
hist_range = [70,110]
bins = hist_range[1]-hist_range[0]


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
def process_events_incremental(reader, max_events=1000, prominence=100, pre_points=20, post_points=50, channel_index = 0):
    counts = np.zeros(bins)
    event_count = 0
    while True:
        event = reader.get_event()
        if event is None:
            break

        y_data = event[channel_index]
        peaks, _ = find_peaks(y_data, prominence=prominence)

        dist = []

        baseline = []

        try:
            if len(peaks)>1:
                for i in range(len(peaks)-1):
                    dist.append(peaks[i+1]-peaks[i])
                dist.append(len(y_data)-peaks[-1])

                for k in range(len(peaks)):
                    if dist[k]>500:
                        if k!=len(peaks)-1:
                            baseline, bin_edges = np.histogram(y_data[peaks[k]+50:peaks[k+1]-50],bins = bins, range =hist_range) 
                        else:                 
                            baseline, bin_edges = np.histogram(y_data[peaks[k]+50:],bins = bins, range =hist_range) 
                        counts= counts + baseline
        except KeyboardInterrupt:
            sys.exit("KeyboardInterrupt")
        except:
            print("error")
        
        
        if event_count%1000==0:
            print(event_count)
        event_count+=1

    return counts, bin_edges


# Usage
if __name__ == "__main__":
    SAVE_PATH = os.path.join(pp.folder_btest,'save_figures',"histos")


    for file in file_list:    

        print(file)
        os.makedirs(SAVE_PATH, exist_ok=True)



        file_name = os.path.join(pp.folder_btest,file)
        for channel in channel_list:
            reader = SuperMUSRBinaryWave()
            reader.load_file(file_name)

            max_events = 10000000000000000000 # maximum n of events 

            counts, bin_edges = process_events_incremental(
                reader,
                max_events=max_events,
                prominence=20,
                pre_points=20,
                post_points=320,
                channel_index = channel
            )

            binscenters = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])

            # plt.plot(binscenters,counts)
            # plt.show()

            data = np.c_[ counts, binscenters ]
            np.savetxt(os.path.join(SAVE_PATH, 'hist_data'+file[:-4]+'.txt'), data, header = file[:-4], delimiter=",")

            

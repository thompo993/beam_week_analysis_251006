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

file_name = "horizontal_20mCables_TestingLD_10OhmA_20251014_.npy"
current_date_time = datetime.today().strftime('%y%m%d_%H%M%S')

SAVE_PATH = os.path.join('save_figures',  file_name[:-4] + "_" +current_date_time)
os.makedirs(SAVE_PATH, exist_ok=True)
BINNING = 8 #set a power of 2 or 0 if you don't want binning
SAVE_CSV = False
norm_binning = False  
xlim = [None,100]
save = "bins" #options: True, False, "expo", "bins"
show = False
pileup_rejection = 1000
size = [15, 10]

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



def exp_decay(x, A, tau, t_max, offset):
    """ Model of exponential decay with time delay t_max """
    return A * np.exp(-(x - t_max) / tau) + offset


# FUnction to calculate rising time (10%-90%) with linear fit 
def calculate_rise_time_with_fit(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)

    # Calculates values for 10% and 90% of the peak
    ten_percent = min_val + 0.1 * (max_val - min_val)
    ninety_percent = min_val + 0.9 * (max_val - min_val)

    # Finds the indices for 10% and 90%
    try:
        ten_percent_index = np.where(signal >= ten_percent)[0][0]
        ninety_percent_index = np.where(signal >= ninety_percent)[0][0]

        # Extracts values for 10% and 90%
        x_rising = np.arange(ten_percent_index, ninety_percent_index + 1)
        y_rising = signal[ten_percent_index:ninety_percent_index + 1]

        # linear fit
        slope, intercept, _, _, _ = linregress(x_rising, y_rising)

        # Calculates time corresponding to 10% and 90% via the linear fit equation
        t_10 = (ten_percent - intercept) / slope
        t_90 = (ninety_percent - intercept) / slope

        rise_time = t_90 - t_10  
    except IndexError:
        rise_time = None  # Not enough indices to calculate risetime 

    return rise_time, (t_10, t_90), (x_rising, y_rising, slope, intercept)

def calculate_tau(signal, max_index):
    x_data = np.arange(len(signal))

    # Shift of the signal to start the fit from the maximum
    x_data_fit = x_data[max_index:]
    signal_fit = signal[max_index:]
    

    # initial parameters for the fit
    A_initial = np.max(signal)  # Signal maximum
    tau_initial = 5  # Initial time constant
    t_max_initial = max_index  # Initial delay = index of maximum of the signal
    offset_initial = np.min(signal)  # minimum offset 

    # fit of the descending section
    try:
        popt, _ = curve_fit(exp_decay, x_data_fit, signal_fit, p0=[A_initial, tau_initial, t_max_initial, offset_initial])
        A, tau, t_max, offset = popt
    except RuntimeError:
        tau, t_max = None, None  # fit not completed

    return tau, t_max, popt  # returns parameters of the fit

# function to update incremental average of signals
def update_average_signal(current_average, new_signal, event_count):
    if current_average is None:
        # if firts signal, change the initial average to actual signal
        return new_signal
    else:
        # updates incremental average
        return (current_average * (event_count - 1) + new_signal) / event_count

# function to process and update the average of exponential signals
def process_events_incremental(reader, max_events=1000, amplitude_range=(1800, 2200), prominence=100, pre_points=20, post_points=50, channel_index = 0):
    average_signal = None
    event_count = 0

    while True:
        event = reader.get_event()
        if event is None or event_count >= max_events:
            break

        y_data = event[channel_index]
        
        # find peaks within specified limits
        peaks_norejection, _ = find_peaks(y_data, height=amplitude_range, prominence=prominence)

        peaks, _ = find_peaks(y_data, distance = pileup_rejection, height=amplitude_range, prominence=prominence)
            
        event_missed = len(peaks_norejection)-len(peaks)

        for peak in peaks:
            # make sure there are enough points around the peak
            if peak - pre_points >= 0 and peak + post_points < len(y_data):
                # extract the segment around the peak
                segment = y_data[peak - pre_points : peak + post_points + 1]
                
                # updates incremental average
                event_count += 1
                average_signal = update_average_signal(average_signal, segment, event_count)

    return average_signal, event_count, event_missed


def plot_exponential_fit_and_tau(signal, tau, t_max, popt):
    x_data = np.arange(len(signal))

    # take only data after the maximum
    x_data_fit = x_data[int(t_max):]
    signal_fit = signal[int(t_max):]

    # exponential function
    exp_fit = exp_decay(x_data_fit, *popt)

    # Plot average signal
    plt.plot(signal, label="Avg signal", color='blue')

    # plot exponential fit after the maximum
    plt.plot(x_data_fit, exp_fit, 'r--', label=f"Exp fit (tau={tau:.3f})")

    # calculates tangent 
    slope = -popt[0] / tau  # derivative of exponential
    tangent_line = popt[0] + slope * (x_data_fit - x_data_fit[0])  # tangent

    # finds the intercept with x axis
    intercept_idx = np.where(tangent_line <= 0)[0][0] if np.any(tangent_line <= 0) else len(tangent_line) - 1
    tangent_line[intercept_idx:] = 0  # tangent is returned to 0 after the calculation


    # Highlight the maximum
    plt.scatter([int(t_max)], [np.max(signal)], color='orange', zorder=5)
    plt.title("Channel %s - %s waves" %(channel, total_events))
    plt.xlabel('ns')
    plt.ylabel('LSB')
    plt.legend()
    plt.xlim(xlim)
    plt.grid(True)
    if save == True or save == "expo":
        plt.savefig(os.path.join(SAVE_PATH, "ch%s_expfit"%channel))    
    if show:
        plt.show()
    plt.clf()
    plt.close()

def binned_average(BIN, norm = True):
    if BIN > 2**12:
            print("Bin number should be less than 2^12")
    else:
        bin_size = 2**12 / BIN
        plt.figure(figsize=size)
        for z in range(BIN):
            print("Processing channel %s, bin %s" %(channel, z))
            reader = SuperMUSRBinaryWave()
            reader.load_file(file_name)
            average_signal_bin, total_events_bin, missed_events_bin = process_events_incremental(
            reader, max_events=max_events, amplitude_range=(z*bin_size, (z+1)*bin_size), prominence=100, pre_points=20, post_points=320, channel_index = channel)
            if norm:
                average_signal_bin = (average_signal_bin- min(average_signal_bin))/(max(average_signal_bin)-min(average_signal_bin))
                average_signal_bin = average_signal_bin[xlim[0]:-xlim[1]]
            else: average_signal_bin = average_signal_bin
            if average_signal_bin is not None:
                plt.plot(average_signal_bin, label="[%s, %s], %s waves, %s missed"%(z*bin_size, (z+1)*bin_size, total_events_bin, missed_events_bin))
            reader.close_file()
        plt.title("Channel %s, number of bins: %s" %(channel, BIN))
        plt.xlabel('ns')
        if norm:
            plt.ylabel('normalised')
        else:
            plt.ylabel('LSB')
        plt.legend()
        plt.xlim(xlim)
        plt.grid(True)
        if save == True or save == "bins":
            plt.savefig(os.path.join(SAVE_PATH, "ch%s_bins"%channel))    
        if show:
            plt.show()
        plt.clf()
        plt.close()

# Usage
if __name__ == "__main__":
    f = open(os.path.join(SAVE_PATH, 'log.txt'), 'w')
    f.write("File name: " + file_name +"\n")


    for channel in range(8):
        reader = SuperMUSRBinaryWave()
        reader.load_file(file_name)

        max_events = 10000000000000000000 # maximum n of events 

  


        if BINNING > 0: 
            binned_average(BINNING, norm_binning)
        else:

                    # Processa i segnali aggiornando la media ad ogni evento
            average_signal, total_events, missed_events = process_events_incremental(
                reader,
                max_events=max_events,
                amplitude_range=(1200, 2200),
                prominence=100,
                pre_points=20,
                post_points=320,
                channel_index = channel
            )

            reader.close_file()

            f.write("\n")
            f.write("Channel %s\n"%channel)      

            #shows average signal if at least one signal was analysed 
            if average_signal is not None:
                plt.figure()
                plt.title("Channel %s - %s waves" %(channel, total_events))
                plt.plot(average_signal, label="Average signal")
                plt.xlabel('ns')
                plt.ylabel('LSB')
                plt.legend()
                plt.xlim(xlim)
                plt.grid(True)
                if save == True:
                    plt.savefig(os.path.join(SAVE_PATH, "ch%s"%channel))
                if show:
                    plt.show()
                plt.clf()
                plt.close()



                # calculate rising time with linear fit 
                rise_time, (t_10, t_90), (x_rising, y_rising, slope, intercept) = calculate_rise_time_with_fit(average_signal)
                # find the index of the peak (maximum)
                max_index = np.argmax(average_signal)

                # show exponential 
                tau, t_max, popt = calculate_tau(average_signal, max_index)

                # save results to log
                f.write(f"Total processed events: {total_events}\n")
                if rise_time is not None:
                    f.write(f"Rising Time (10-90%): {rise_time:.3f} points\n")
                    f.write(f"t_10: {t_10:.3f}, t_90: {t_90:.3f}\n")

                    # show linear fit for rising time
                    plt.figure()
                    plt.title("Channel %s - %s waves" %(channel, total_events))
                    plt.plot(average_signal, label="Rise Time: {:.3f}".format(rise_time))
                    plt.plot(x_rising, slope * x_rising + intercept, 'r--', label="Linear Fit (10-90%)")
                    plt.axvline(t_10, color='green', linestyle='--', label='t_10 (10%)')
                    plt.axvline(t_90, color='orange', linestyle='--', label='t_90 (90%)')
                    plt.xlabel('ns')
                    plt.ylabel('LSB')
                    plt.xlim(xlim)
                    plt.legend()
                    plt.grid(True)
                    if save == True: 
                        plt.savefig(os.path.join(SAVE_PATH, "ch%s_risetime"%channel))
                    if show:
                        plt.show()
                    plt.clf()
                    plt.close()

                    #save the average signal as CSV file
                    if SAVE_CSV:
                        np.savetxt(os.path.join(SAVE_PATH,"ch%s_averagesignal.csv"%channel), average_signal, delimiter=",")

                else:
                    f.write("Rising time was not calculated.\n")

                # Modifica nella sezione di calcolo e visualizzazione di tau
                if tau is not None:
                    f.write(f"Tau: {tau:.3f} temporal unit\n")

                    plot_exponential_fit_and_tau(average_signal, tau, max_index, popt)

                else:
                    f.write("Tau was not calculated\n")

            else:
                f.write("No average signal available\n")
    

    f.close()


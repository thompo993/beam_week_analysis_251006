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
    """ Modello di decadimento esponenziale con ritardo temporale t_max """
    return A * np.exp(-(x - t_max) / tau) + offset


# Funzione per calcolare il rising time (10%-90%) con fit rettilineo
def calculate_rise_time_with_fit(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)

    # Calcola i valori corrispondenti al 10% e al 90% del picco
    ten_percent = min_val + 0.1 * (max_val - min_val)
    ninety_percent = min_val + 0.9 * (max_val - min_val)

    # Trova gli indici corrispondenti al 10% e al 90%
    try:
        ten_percent_index = np.where(signal >= ten_percent)[0][0]
        ninety_percent_index = np.where(signal >= ninety_percent)[0][0]

        # Estrai i punti tra il 10% e il 90%
        x_rising = np.arange(ten_percent_index, ninety_percent_index + 1)
        y_rising = signal[ten_percent_index:ninety_percent_index + 1]

        # Esegui un fit lineare sui punti
        slope, intercept, _, _, _ = linregress(x_rising, y_rising)

        # Calcola i tempi corrispondenti al 10% e al 90% tramite l'equazione del fit lineare
        t_10 = (ten_percent - intercept) / slope
        t_90 = (ninety_percent - intercept) / slope

        rise_time = t_90 - t_10  # Tempo di salita
    except IndexError:
        rise_time = None  # Non ci sono sufficienti punti per calcolare il tempo di salita

    return rise_time, (t_10, t_90), (x_rising, y_rising, slope, intercept)

def calculate_tau(signal, max_index):
    x_data = np.arange(len(signal))

    # Shift del segnale per iniziare il fit dal massimo
    x_data_fit = x_data[max_index:]
    signal_fit = signal[max_index:]

    # Parametri iniziali più accurati per il fit
    A_initial = np.max(signal)  # Il massimo del segnale
    tau_initial = 5  # Una costante di tempo iniziale
    t_max_initial = max_index  # Il ritardo iniziale (indice del massimo)
    offset_initial = np.min(signal)  # L'offset minimo del segnale

    # Fit solo per il tratto discendente
    try:
        popt, _ = curve_fit(exp_decay, x_data_fit, signal_fit, p0=[A_initial, tau_initial, t_max_initial, offset_initial])
        A, tau, t_max, offset = popt
    except RuntimeError:
        tau, t_max = None, None  # Il fit non è riuscito

    return tau, t_max, popt  # Restituiamo anche i parametri del fit

# Funzione per aggiornare la media incrementale dei segnali
def update_average_signal(current_average, new_signal, event_count):
    if current_average is None:
        # Se è il primo evento, il segnale attuale diventa la media iniziale
        return new_signal
    else:
        # Aggiorna la media incrementale
        return (current_average * (event_count - 1) + new_signal) / event_count

# Funzione per processare e aggiornare la media dei segnali esponenziali
def process_events_incremental(reader, max_events=1000, amplitude_range=(1800, 2200), prominence=100, pre_points=20, post_points=50):
    average_signal = None
    event_count = 0
    
    while True:
        event = reader.get_event()
        if event is None or event_count >= max_events:
            break

        y_data = event[0]
        
        # Trova i picchi con altezza tra i limiti specificati
        peaks, _ = find_peaks(y_data, height=amplitude_range, prominence=prominence)

        for peak in peaks:
            # Assicurati che ci siano abbastanza punti prima e dopo il picco
            if peak - pre_points >= 0 and peak + post_points < len(y_data):
                # Estrai il segmento attorno al picco
                segment = y_data[peak - pre_points : peak + post_points + 1]
                
                # Aggiorna la media incrementale
                event_count += 1
                average_signal = update_average_signal(average_signal, segment, event_count)

    return average_signal, event_count


def plot_exponential_fit_and_tau(signal, tau, t_max, popt):
    x_data = np.arange(len(signal))

    # Prendi i dati solo dopo il massimo
    x_data_fit = x_data[int(t_max):]
    signal_fit = signal[int(t_max):]

    # Funzione esponenziale per il fit (a partire dal massimo con i parametri del fit)
    exp_fit = exp_decay(x_data_fit, *popt)

    # Plot del segnale medio
    plt.plot(signal, label="Avg signal", color='blue')

    # Plot del fit esponenziale (solo dopo il massimo)
    plt.plot(x_data_fit, exp_fit, 'r--', label=f"Exp fit (tau={tau:.3f})", color='red')

    # Calcolo della tangente a partire dal massimo
    slope = -popt[0] / tau  # La derivata del decadimento esponenziale
    tangent_line = popt[0] + slope * (x_data_fit - x_data_fit[0])  # Tangente

    # Trova l'intercetta della tangente con l'asse x (dove y=0)
    intercept_idx = np.where(tangent_line <= 0)[0][0] if np.any(tangent_line <= 0) else len(tangent_line) - 1
    tangent_line[intercept_idx:] = 0  # Imposta la tangente a zero dopo l'intercetta


    # Evidenzia il massimo
    plt.scatter([int(t_max)], [np.max(signal)], color='orange', zorder=5)

    plt.xlabel('ns')
    plt.ylabel('LSB')
    plt.legend()
    plt.grid(True)
    plt.show()



# Esempio di utilizzo
if __name__ == "__main__":
    file_name = "waves_setup_cat7_3m_flat_cat8_10m.npy"
    reader = SuperMUSRBinaryWave()
    reader.load_file(file_name)

    max_events = 10000000000000000000 # Limite massimo di eventi da elaborare

    # Processa i segnali aggiornando la media ad ogni evento
    average_signal, total_events = process_events_incremental(
        reader,
        max_events=max_events,
        amplitude_range=(1800, 2200),
        prominence=100,
        pre_points=20,
        post_points=320
    )

    reader.close_file()

    # Mostra il segnale medio solo se è stato processato almeno un evento
    if average_signal is not None:
        plt.show()

        # Calcola il rising time con fit lineare e tau
        rise_time, (t_10, t_90), (x_rising, y_rising, slope, intercept) = calculate_rise_time_with_fit(average_signal)
        # Trova l'indice del massimo (picco)
        max_index = np.argmax(average_signal)

        # Visualizza il fit esponenziale e la tangente a partire dal massimo
        tau, t_max, popt = calculate_tau(average_signal, max_index)

        # Stampa i risultati
        print(f"Numero totale di eventi processati: {total_events}")
        if rise_time is not None:
            print(f"Rising Time (10-90%): {rise_time:.3f} punti")
            print(f"t_10: {t_10:.3f}, t_90: {t_90:.3f}")

            # Mostra il fit lineare sul tratto di rising
            plt.figure()
            plt.plot(average_signal, label="Rise Time: {:.3f}".format(rise_time))
            plt.plot(x_rising, slope * x_rising + intercept, 'r--', label="Linear Fit (10-90%)")
            plt.axvline(t_10, color='green', linestyle='--', label='t_10 (10%)')
            plt.axvline(t_90, color='orange', linestyle='--', label='t_90 (90%)')
            plt.xlabel('ns')
            plt.ylabel('LSB')
            
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Non è stato possibile calcolare il rising time.")

        # Modifica nella sezione di calcolo e visualizzazione di tau
        if tau is not None:
            print(f"Tau: {tau:.3f} unità temporali")

            plot_exponential_fit_and_tau(average_signal, tau, max_index, popt)

        else:
            print("Non è stato possibile calcolare la tau.")

    else:
        print("Nessun segnale medio disponibile.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
from colorama import Fore, Style, init, Back
import csv
from calibration_settings import *
plt.rcParams.update({'font.size': 20})
V = V1


RL = ["RIGHT", "LEFT"]
#P = [1]
P = ENABLED_CH


offset_x = 20
width_x= 650*2
pe_target = PE_TARGET

source_folder = "run_hv"
output_folder = 'run_fit'

# V = np.sort(V)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)  
    
# Definizione della funzione composta da più gaussiane
def multiple_gaussians(x, *params):
    y = np.zeros_like(x, dtype=np.float64)  # Inizializza y come float64
    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i+1]
        sigma = params[i+2]
        y += amplitude * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
    return y


# Funzione per trovare la valle entro ±30 punti
def find_valley_near_peak(spectrum, peak_idx, window=30):
    # Cerca a sinistra
    left_idx = max(0, peak_idx - window)
    right_idx = min(len(spectrum) - 1, peak_idx + window)
    
    width = right_idx - left_idx
    
    # Trova la valle a sinistra e destra
    left_valley = np.min(spectrum[left_idx:peak_idx])
    right_valley = np.min(spectrum[peak_idx:right_idx])
    
    # Ritorna la valle più profonda (valore minore)
    return min(left_valley, right_valley), width


def removeTriggerFakePeak(peaks_a):

    # Lista per mantenere i picchi validi
    valid_peaks = []

    # Filtra i picchi
    for peak in peaks_a:
        # Controlla se ci sono almeno 5 campioni a sinistra del picco
        if peak >= 5:
            # Valore a sinistra di 5 campioni rispetto al picco
            left_value = spectrum[peak - 5]
            # Valore del picco
            peak_value = spectrum[peak]
            
            # Se il valore a sinistra è inferiore al 10% dell'altezza del picco, lo eliminiamo
            if left_value >= 0.1 * peak_value:
                valid_peaks.append(peak)
                
    # Restituisci i picchi validi e le loro proprietà
    return np.array(valid_peaks)
                
def FitAndSave(spectrum, file_output, channel, current_voltage):
    
    # Trova il picco massimo
    max_peak_idx = np.argmax(spectrum)
    max_peak_value = spectrum[max_peak_idx]

    print(f"Max peak value: {max_peak_value}")
    print(f"Max peak index: {max_peak_idx}")

    # Trova la valle più vicina
    valley_value, ww = find_valley_near_peak(spectrum, max_peak_idx)


    # Calcola la prominenza automatica come il 10% della differenza tra massimo e valle
    prominence_auto = 0.05 * (max_peak_value - valley_value)

    # Trova i picchi nello spettro
    x_data = np.arange(len(spectrum))
    peaks_a, _ = find_peaks(spectrum, height=0.05 * max(spectrum), prominence=prominence_auto, distance=4)  # Trova picchi significativi
    peaks_a = removeTriggerFakePeak(peaks_a)
    peaks_a_diff = np.diff(peaks_a) 

    mean_diff = np.mean(peaks_a_diff)

    print("Raw mean diff: ", mean_diff)
    prominence_auto = 0.1 * (max_peak_value - valley_value)


    peaks, _ = find_peaks(spectrum, height=0.03 * max(spectrum), prominence=prominence_auto, distance=mean_diff*0.85)  # Trova picchi significativi

    print(Fore.GREEN +f"Peaks found: {peaks}" + Style.RESET_ALL)
    # Estrai i parametri iniziali per ogni picco trovato
    initial_guesses = []

    peaks_a_diff = np.diff(peaks) 

    sigma_v = np.mean(peaks_a_diff)/4.6

    for peak in peaks:
        amplitude = spectrum[peak]  # Altezza del picco
        mean = peak  # Posizione del picco
        sigma = sigma_v  # Larghezza stimata, dovrai adattarla ai tuoi dati reali
        initial_guesses.extend([amplitude, mean, sigma])

    # Fit dei dati
    params, covariance = curve_fit(multiple_gaussians, x_data, spectrum, p0=initial_guesses)#, method="dogbox")

    # Crea un asse x 10 volte più fine per il plot
    x_fine = np.linspace(0, len(spectrum) - 1, 10 * len(spectrum))

    # Calcola il fit sul nuovo asse x più fine
    fit_result_fine = multiple_gaussians(x_fine, *params)



    # Estrai le posizioni dei picchi (mean)
    mean_positions = np.array([params[i+1] for i in range(0, len(params), 3)])

    # Calcola le differenze tra ogni coppia di picchi consecutivi
    diffs = np.diff(mean_positions)

    # Funzione per eseguire il sigma-clipping
    def sigma_clipping(data, sigma=1, max_iterations=10):
        for _ in range(max_iterations):
            mean = np.mean(data)
            std = np.std(data)
            filtered_data = data[np.abs(data - mean) <= sigma * std]
            # Se il nuovo array è uguale a quello precedente, possiamo fermarci
            if len(filtered_data) == len(data):
                break
            data = filtered_data
        return np.mean(filtered_data), filtered_data

    # Esegui il sigma clipping sulle differenze tra picchi
    final_mean_diff, filtered_diffs = sigma_clipping(diffs, sigma=1)

    # Stampa la media finale
    print(f"Channel: {channel}, Voltage: {current_voltage},  Single photon LSB: {final_mean_diff}")
    # Plot dei dati e del fit

    maximum = np.argmax(spectrum)

    plt.figure(figsize = [14,10])
    plt.plot(x_data-maximum, spectrum, label='Data')
    plt.plot(x_fine-maximum, fit_result_fine, label='Fit')
    plt.scatter(peaks-maximum, spectrum[peaks], color='red', zorder=5, label='Peaks')  # Mostra i picchi trovati
    plt.title(f'Single photon height:{final_mean_diff:.2f} LSB')
    plt.xlabel('LSB')
    plt.ylabel('Counts')
    plt.xlim(-250,400)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # Salva la figura su file senza aprirla
    plt.savefig(file_output, format='png')
    plt.close()  # Chiude la figura senza mostrarla
    
    peaks = removeTriggerFakePeak(peaks)
    peaks_diff = np.diff(peaks) 

    mean_diff = np.mean(peaks_diff)
    
    if (final_mean_diff - mean_diff) > mean_diff*0.25:
        print(Fore.RED + "Errore nel fit, uso il valore medio" + Style.RESET_ALL )
        return mean_diff
    else:
        return final_mean_diff
VOLTAGE_ARRAY = [0 for i in range(16)]

# Loop over P to generate data and graphs
for xp_in in P:
    voltages_right = []
    pe_values_right = []
    voltages_left = []
    pe_values_left = []

    for x_rl in RL:
        voltages = []
        pe_values = []
        
        for x_v in V:
            channel = x_rl + "_" + str(xp_in)
            output_filename = f'{channel}_{int(1000.0*x_v)}.png'
            output_path = os.path.join(output_folder, output_filename)

            file_to_load = f'{source_folder}/_{x_rl}_PULSER_{xp_in}_{str(x_v)}_spectra_A.csv'
            print(Back.MAGENTA + f'Load file: {file_to_load}'+ Style.RESET_ALL)
            # Read the CSV
            df = pd.read_csv(file_to_load)

            # Extract the first column and convert it to a NumPy array
            spectrum = df.iloc[:, xp_in].values[offset_x:offset_x+width_x]

            try:
                # Execute FitAndSave function and collect data (returns single value)
                pe = FitAndSave(spectrum, output_path, channel, x_v)

                # Save results for the graph
                voltages.append(x_v)
                pe_values.append(pe)

                print("P", xp_in, "RL", x_rl, "V", x_v, "PE", pe)
            except Exception as e:
                print(Fore.RED + f"Fit error for {x_rl} - P{xp_in} - V{x_v}: {e}" + Style.RESET_ALL)
        
        # Add data to lists for RIGHT and LEFT
        if x_rl == "RIGHT":
            voltages_right = voltages
            pe_values_right = pe_values
        else:
            voltages_left = voltages
            pe_values_left = pe_values

        # Original data
        p = np.array(pe_values)
        v = np.array(voltages)

        # 1) Mask to remove NaN/Inf
        mask_ok = np.isfinite(p) & np.isfinite(v)
        p0 = p[mask_ok]
        v0 = v[mask_ok]

        # 2) Calculate first and third quartile and IQR
        q1, q3 = np.percentile(v0, [25, 75])
        iqr = q3 - q1

        # 3) Tukey thresholds: outside [q1 - 1.5·IQR, q3 + 1.5·IQR] → outlier
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # 4) Final mask: keep only "normal" values
        mask_iqr = (v0 >= lower) & (v0 <= upper)
        p_clean = p0[mask_iqr]
        v_clean = v0[mask_iqr]

        # Now p_clean and v_clean are free of NaN and outliers
        # Fit on these values:
        coeffs = np.polyfit(p_clean, v_clean, 1)
        fit_line = np.poly1d(coeffs)

        print (coeffs)

        # Fit equation
        slope, intercept = coeffs
        equation = f'V = {slope:.2f} * PE + {intercept:.2f}'
        
        # Calculate V at PE=0 (intercept)
        v_at_pe_zero = intercept

        v_target = fit_line(pe_target)

        # Create individual graph for the P and RL combination
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(pe_values, voltages, marker='o', linestyle='None', color='b', label='Data')
        ax.plot(pe_values, fit_line(pe_values), linestyle='--', color='r', label=f'Fit: {equation}')

        # Add cursors to indicate V corresponding to desired PE
        ax.axhline(y=v_target, color='g', linestyle=':', label=f'V target for PE={pe_target}')
        ax.axvline(x=pe_target, color='g', linestyle=':')

        # Show V and PE values on respective axes
        ax.text(pe_target, v_target, f'PE={pe_target}, V={v_target:.2f}', color='g', fontsize=12, verticalalignment='bottom')

        # Add labels and title with V(PE=0)
        ax.set_xlabel('PE')
        ax.set_ylabel('V')
        ax.set_title(f'PE vs V graph for {x_rl} - P{xp_in}\nV(PE=0) = {v_at_pe_zero:.2f}')
        ax.grid(True)

        # Activate interactive cursors
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

        # Add legend
        ax.legend()

        # Save graph to PNG file with 2048x2048 resolution
        graph_filename = f'{x_rl}_P{xp_in}_PE_vs_V_with_fit.png'
        plt.savefig(os.path.join(output_folder, graph_filename), dpi=204.8)
        plt.close()

        print(f'Graph saved as: {graph_filename}')
    
    # Create additional graph with both RIGHT and LEFT
    fig, ax = plt.subplots(figsize=(10, 10))

    def remove_outliers(pe_values, voltages, threshold=1):
        
        # Convert data to NumPy arrays (if not already)
        pe_values = np.array(pe_values)
        voltages = np.array(voltages)
        
        # Filter points with pe_values < 5 or pe_values > 55
        mask_pe = (pe_values >= 5) & (pe_values <= 55)
        pe_values = pe_values[mask_pe]
        voltages = voltages[mask_pe]
        
        # If either is already empty, exit immediately
        if pe_values.size == 0 or voltages.size == 0:
            return np.array([]), np.array([])
            
        # First fit
        coeffs = np.polyfit(pe_values, voltages, 1)
        fit_line = np.poly1d(coeffs)
        
        # Calculate estimates based on fit
        voltages_fit = fit_line(pe_values)
        
        # Calculate residuals (distance between real value and fit)
        residuals = voltages - voltages_fit
        
        # Calculate standard deviation of residuals
        sigma = np.std(residuals)
        
        # Keep only points within threshold sigma
        mask = np.abs(residuals) < threshold * sigma
        
        # Apply mask to data
        return pe_values[mask], voltages[mask]
    
    # Fit RIGHT with outlier removal
    pe_values_right_filtered, voltages_right_filtered = remove_outliers(pe_values_right, voltages_right)
    print(pe_values_right, voltages_right)
    print(pe_values_right_filtered, voltages_right_filtered)
    if pe_values_right_filtered.size == 0 or voltages_right_filtered.size == 0:
        continue   
    
    # New fit after outlier removal
    coeffs_right = np.polyfit(pe_values_right_filtered, voltages_right_filtered, 1)
    fit_line_right = np.poly1d(coeffs_right)
    v_target_right = fit_line_right(pe_target)
    v_at_pe_zero_right = coeffs_right[1]  # Intercept for RIGHT

    # Fit LEFT with outlier removal
    pe_values_left_filtered, voltages_left_filtered = remove_outliers(pe_values_left, voltages_left)

    # New fit after outlier removal
    coeffs_left = np.polyfit(pe_values_left_filtered, voltages_left_filtered, 1)
    fit_line_left = np.poly1d(coeffs_left)
    v_target_left = fit_line_left(pe_target)
    v_at_pe_zero_left = coeffs_left[1]  # Intercept for LEFT
    

    # Plot RIGHT
    ax.plot(pe_values_right, voltages_right, marker='o', linestyle='None', color='b', label='RIGHT Data')
    ax.plot(pe_values_right, fit_line_right(pe_values_right), linestyle='--', color='r', label='RIGHT Fit')

    # Plot LEFT
    ax.plot(pe_values_left, voltages_left, marker='o', linestyle='None', color='c', label='LEFT Data')
    ax.plot(pe_values_left, fit_line_left(pe_values_left), linestyle='--', color='m', label='LEFT Fit')

    # Add cursors to indicate V corresponding to target PE
    ax.axhline(y=v_target_right, color='g', linestyle=':', label=f'RIGHT V target for PE={pe_target}')
    ax.axhline(y=v_target_left, color='y', linestyle=':', label=f'LEFT V target for PE={pe_target}')
    ax.axvline(x=pe_target, color='g', linestyle=':', label=f'PE target={pe_target}')

    # Show V and PE values for RIGHT and LEFT on respective axes
    ax.text(pe_target, v_target_right, f'PE={pe_target}, RIGHT V={v_target_right:.2f}', color='g', fontsize=12, verticalalignment='bottom')
    ax.text(pe_target, v_target_left, f'PE={pe_target}, LEFT V={v_target_left:.2f}', color='y', fontsize=12, verticalalignment='bottom')

    # Add labels and title with V(PE=0) for both sides
    ax.set_xlabel('PE')
    ax.set_ylabel('V')
    ax.set_title(f'Combined PE vs V graph for RIGHT and LEFT - P{xp_in}\n' + 
                 f'RIGHT: V(PE=0) = {v_at_pe_zero_right:.2f} | LEFT: V(PE=0) = {v_at_pe_zero_left:.2f}')
    ax.grid(True)

    # Add legend
    ax.legend()

    # Save combined graph
    combined_graph_filename = f'Combined_RIGHT_LEFT_P{xp_in}_PE_vs_V_with_fit.png'
    plt.savefig(os.path.join(output_folder, combined_graph_filename), dpi=204.8)
    plt.close()

    print(f'Combined graph saved as: {combined_graph_filename}')
    
    VOLTAGE_ARRAY[MAP_LEFT[xp_in]] = v_target_left
    VOLTAGE_ARRAY[MAP_RIGHT[xp_in]] = v_target_right

for i in range(0, len(VOLTAGE_ARRAY)):
    VOLTAGE_ARRAY[i] = VOLTAGE_ARRAY[i] * V_cM - V_cQ


print(VOLTAGE_ARRAY)

with open(f'run/hv_calibration_for_pe_{pe_target}.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(VOLTAGE_ARRAY)
# # Loop su P, RL, V per generare i dati e i grafici
# for xp_in in P:
#     for x_rl in RL:
#         voltages = []
#         pe_values = []
        
#         for x_v in V:
#             channel = x_rl + "_" + str(xp_in)
#             output_filename = f'{channel}_{int(1000.0*x_v)}.png'
#             output_path = os.path.join(output_folder, output_filename)

#             file_to_load = f'calib/_{x_rl}_PULSER_{xp_in}_{str(x_v)}_spectra_A.csv'
#             print(f'Load file: {file_to_load}')
#             # Leggi il CSV
#             df = pd.read_csv(file_to_load)

#             # Estrai la prima colonna e convertila in un array NumPy
#             spectrum = df.iloc[:, xp_in].values[0:350]


#             # Esegui la funzione FitAndSave e raccogli i dati
#             pe = FitAndSave(spectrum, output_path, channel, x_v)

#             # Salva i risultati per il grafico
#             voltages.append(x_v)
#             pe_values.append(pe)

#             print("P", xp_in, "RL", x_rl, "V", x_v, "PE", pe)
            
            
#         # Fit rettilineo (polinomio di grado 1)
#         coeffs = np.polyfit(pe_values, voltages, 1)
#         fit_line = np.poly1d(coeffs)

#         # Equazione del fit
#         slope, intercept = coeffs
#         equation = f'V = {slope:.2f} * PE + {intercept:.2f}'

#         v_target = fit_line(pe_target)

#         # Creare un grafico per la combinazione P e RL
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.plot(pe_values, voltages, marker='o', linestyle='-', color='b', label='Dati')
#         ax.plot(pe_values, fit_line(pe_values), linestyle='--', color='r', label=f'Fit: {equation}')

#         # Aggiungi cursori per indicare la V corrispondente al PE desiderato
#         ax.axhline(y=v_target, color='g', linestyle=':', label=f'V target per PE={pe_target}')
#         ax.axvline(x=pe_target, color='g', linestyle=':')

#         # Mostra il valore di V e PE sui rispettivi assi
#         ax.text(pe_target, v_target, f'PE={pe_target}, V={v_target:.2f}', color='g', fontsize=12, verticalalignment='bottom')

#         # Aggiungi etichette e titolo
#         ax.set_xlabel('PE')
#         ax.set_ylabel('V')
#         ax.set_title(f'Grafico PE vs V per {x_rl} - P{xp_in}')
#         ax.grid(True)

#         # Attiva i cursori interattivi
#         cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

#         # Aggiungi la legenda
#         ax.legend()

#         # Salva il grafico in un file PNG con risoluzione 2048x2048
#         graph_filename = f'{x_rl}_P{xp_in}_PE_vs_V_with_fit.png'
#         plt.savefig(os.path.join(output_folder, graph_filename), dpi=204.8)  # Imposta dpi per ottenere 2048x2048 pixel
#         plt.close()  # Chiudi la figura per evitare problemi con le figure successive

#         print(f'Grafico salvato come: {graph_filename}')
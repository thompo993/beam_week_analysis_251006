import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filepath = r"File Path Here"
savepath = r"Save Path Here"

# Read CSV
df = pd.read_csv(filepath, on_bad_lines='skip', delimiter=",")

# Prepare data
lsb = np.array(df.index)  # Convert index to numpy array
v_55 = df['55V Single Photon Height 26.84 LSB']
v_56 = df['56V Single Photon Height 34.27 LSB']
v_57 = df['57V Single Photon Height 41.60 LSB']
v_58 = df['58V Single Photon Height 48.82 LSB']
v_59 = df['59V Single Photon Height 56.07 LSB']

# Function to find peak position
def find_peak(data):
    """Find the index of the maximum value (peak)"""
    return np.argmax(data)

# Find peaks for each dataset
peak_55 = find_peak(v_55)
peak_56 = find_peak(v_56)
peak_57 = find_peak(v_57)
peak_58 = find_peak(v_58)
peak_59 = find_peak(v_59)

# Create centered x-axis for each dataset (relative to peak)
centered_55 = lsb - peak_55
centered_56 = lsb - peak_56
centered_57 = lsb - peak_57
centered_58 = lsb - peak_58
centered_59 = lsb - peak_59

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot data with centered x-axes
ax.plot(centered_55, v_55, label="55V Single Photon Height 26.84 LSB", linewidth=1)
ax.plot(centered_56, v_56, label='56V Single Photon Height 34.27 LSB', linewidth=1)
ax.plot(centered_57, v_57, label='57V Single Photon Height 41.60 LSB', linewidth=1)
ax.plot(centered_58, v_58, label='58V Single Photon Height 48.82 LSB', linewidth=1)
ax.plot(centered_59, v_59, label='59V Single Photon Height 56.07 LSB', linewidth=1)

# Format plot
ax.set_xlabel("Amplitude relative to peak [LSB]", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.set_title("Layered overlay of MPS at different ∆LSB", fontsize=20)
ax.legend(loc='best', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, alpha=0.3)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlim(-250,400)

plt.tight_layout()
plt.savefig(savepath, dpi=300)
plt.show()

plt.tight_layout()
plt.show()
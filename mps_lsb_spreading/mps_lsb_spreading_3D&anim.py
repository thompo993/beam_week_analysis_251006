import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


filepath = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\251006_beam_analysis\plots_data_for_japan_conference_251024\spreading_lsb_mCA_LHS-CH6.csv"
savepath = r"\\isis\Shares\DigitalMuons\supermusr_stavetesting_beam_week_251006\japan_conference_plots_251024\dlsb_spreading_centred\powerpoint_animation\59V.png"

savepath_3d = r"\\isis\Shares\DigitalMuons\supermusr_stavetesting_beam_week_251006\japan_conference_plots_251024\dlsb_spreading_centred\test_3d_and_anim\spreading_lsb_mCA_LHS_CH6_3d.png"  # Path for 3D plot
savepath_anim = r"\\isis\Shares\DigitalMuons\supermusr_stavetesting_beam_week_251006\japan_conference_plots_251024\dlsb_spreading_centred\test_3d_and_anim\spreading_lsb_mCA_LHS_CH6_mp4.mp4" # Path for animation (e.g., .gif or .mp4)

# Read CSV
df = pd.read_csv(filepath, on_bad_lines='skip', delimiter=",")

# Prepare data
lsb = np.array(df.index)
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

# Store all data for easy iteration
voltages = [55, 56, 57, 58, 59]
datasets = [v_55, v_56, v_57, v_58, v_59]
centered_data = [centered_55, centered_56, centered_57, centered_58, centered_59]
lsb_heights = [26.84, 34.27, 41.60, 48.82, 56.07]

# ============================================
# 1. 3D PLOT
# ============================================
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot each voltage as a separate line in 3D space
for i, (voltage, centered_x, data) in enumerate(zip(voltages, centered_data, datasets)):
    # Create y-coordinate (voltage level)
    y = np.full_like(centered_x, voltage)
    ax.plot(centered_x, y, data, label=f'{voltage}V (LSB: {lsb_heights[i]})', linewidth=2)

# Format 3D plot
ax.set_xlabel("Amplitude relative to peak [LSB]", fontsize=16, labelpad=10)
ax.set_ylabel("Voltage [V]", fontsize=16, labelpad=10)
ax.set_zlabel("Counts", fontsize=16, labelpad=10)
ax.set_title("3D Overlay of MPS at different voltages", fontsize=20, pad=20)
ax.legend(loc='best', fontsize=12)
ax.set_xlim(-250, 400)
ax.view_init(elev=20, azim=45)  # Set viewing angle
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(savepath_3d, dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 2. ANIMATION
# ============================================
fig, ax = plt.subplots(figsize=(14, 10))

# Initialize empty line
line, = ax.plot([], [], linewidth=2.5)
title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=20)

# Format plot
ax.set_xlabel("Amplitude relative to peak [LSB]", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, alpha=0.3)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlim(-250, 400)

# Set y-limits based on data range
y_max = max([data.max() for data in datasets])
ax.set_ylim(0, y_max * 1.1)

def init():
    """Initialize animation"""
    line.set_data([], [])
    title.set_text('')
    return line, title

def animate(frame):
    """Animation function"""
    # Interpolate between datasets
    n_datasets = len(datasets)
    frames_per_transition = 30
    total_frames = (n_datasets - 1) * frames_per_transition
    
    if frame < total_frames:
        # Calculate which transition we're in
        transition_idx = frame // frames_per_transition
        progress = (frame % frames_per_transition) / frames_per_transition
        
        # Interpolate between current and next dataset
        current_data = datasets[transition_idx]
        next_data = datasets[transition_idx + 1]
        current_x = centered_data[transition_idx]
        next_x = centered_data[transition_idx + 1]
        
        # Linear interpolation
        interp_data = current_data * (1 - progress) + next_data * progress
        interp_x = current_x * (1 - progress) + next_x * progress
        
        # Interpolate voltage and LSB for title
        current_v = voltages[transition_idx]
        next_v = voltages[transition_idx + 1]
        interp_v = current_v * (1 - progress) + next_v * progress
        
        current_lsb = lsb_heights[transition_idx]
        next_lsb = lsb_heights[transition_idx + 1]
        interp_lsb = current_lsb * (1 - progress) + next_lsb * progress
        
        line.set_data(interp_x, interp_data)
        title.set_text(f'Voltage: {interp_v:.1f}V | Single Photon Height: {interp_lsb:.2f} LSB')
    else:
        # Hold on final frame
        line.set_data(centered_data[-1], datasets[-1])
        title.set_text(f'Voltage: {voltages[-1]}V | Single Photon Height: {lsb_heights[-1]} LSB')
    
    return line, title

# Create animation
n_frames = (len(datasets) - 1) * 30 + 30  # 30 frames per transition + 30 frames hold at end
anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=50, blit=True, repeat=True)

# # Save animation
# # For GIF: use writer='pillow'
# # For MP4: use writer='ffmpeg'
# if savepath_anim.endswith('.gif'):
#     anim.save(savepath_anim, writer='pillow', fps=20, dpi=150)
# elif savepath_anim.endswith('.mp4'):
#     anim.save(savepath_anim, writer='ffmpeg', fps=20, dpi=150)

plt.tight_layout()
plt.show()

print("3D plot and animation created successfully!")
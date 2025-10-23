import pandas as pd
import matplotlib.pyplot as plt

filepath = "file path here"

# Read CSV and skip bad lines
df = pd.read_csv(filepath, on_bad_lines='skip', delimiter=",")
print(df)

# Now the first column is the index
x = df.index                  # first column (used as index)
y = df.iloc[:,0]             # second column (first data column)

# ============================================
# CUSTOMIZATION PARAMETERS
# ============================================

# Figure settings
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6
DPI = 250

# Axis limits
Y_MIN = 0
Y_MAX = 300
X_MIN = 0
X_MAX = 4096

# Log scale settings
X_LOG_SCALE = False  # Set to True for logarithmic x-axis
Y_LOG_SCALE = False  # Set to True for logarithmic y-axis

# Labels and title
X_LABEL = "Amplitude [LSB]"
Y_LABEL = "Counts"
TITLE = "Pulse Height Spectra - PE=25"  
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
TICK_FONTSIZE = 14

# Plot line settings
LINE_COLOR = 'blue'
LINE_WIDTH = 1
LINE_STYLE = '-'  # Options: '-', '--', '-.', ':'
MARKER = None  # Options: 'o', 's', '^', 'x', '+', etc. or None
MARKER_SIZE = 4

# Grid settings
GRID_ENABLED = True
GRID_COLOR = 'gray'
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5
WHICH_GRID = 'both'  # Options: 'major', 'minor', 'both'

# Minor grid settings (optional)
MINOR_GRID_ENABLED = False

# Tight layout
TIGHT_LAYOUT = True

# Save figure settings
SAVE_FIGURE = False  # Set to True to save the figure
SAVE_PATH = "output_plot.png"  # Path where figure will be saved
SAVE_DPI = 300  # DPI for saved figure (can be different from display DPI)
SAVE_FORMAT = 'png'  # Options: 'png', 'pdf', 'svg', 'jpg', etc.
SAVE_BBOX_INCHES = 'tight'  # Options: 'tight' or None
SAVE_TRANSPARENT = False  # Set to True for transparent background

# ============================================
# CREATE PLOT
# ============================================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

# Set log scales
if X_LOG_SCALE:
    ax.set_xscale('log')
if Y_LOG_SCALE:
    ax.set_yscale('log')

# Set axis limits
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlim(X_MIN, X_MAX)

# Plot data
ax.plot(x, y, 
        color=LINE_COLOR, 
        linewidth=LINE_WIDTH, 
        linestyle=LINE_STYLE,
        marker=MARKER,
        markersize=MARKER_SIZE)

# Set labels
ax.set_xlabel(X_LABEL, fontsize=LABEL_FONTSIZE)
ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONTSIZE)
if TITLE:
    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)

# Set tick font sizes
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

# Configure grid
if GRID_ENABLED:
    ax.grid(True, 
            which=WHICH_GRID,
            color=GRID_COLOR, 
            alpha=GRID_ALPHA, 
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH)
    
    # Enable minor ticks if using minor grid
    if MINOR_GRID_ENABLED or WHICH_GRID in ['minor', 'both']:
        ax.minorticks_on()

# Apply tight layout
if TIGHT_LAYOUT:
    plt.tight_layout()

# Save figure if enabled
if SAVE_FIGURE:
    plt.savefig(SAVE_PATH, 
                dpi=SAVE_DPI, 
                format=SAVE_FORMAT,
                bbox_inches=SAVE_BBOX_INCHES,
                transparent=SAVE_TRANSPARENT)
    print(f"Figure saved to: {SAVE_PATH}")

plt.show()
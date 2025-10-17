import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Path to your CSV file
filename = r"\\isis\shares\Detectors\Ben Thompson 2025-2026\Ben Thompson 2025-2025 Shared\Labs\251006_beam_analysis\SiPM_Vbr_log_251016.csv"

# Read the CSV safely
df = pd.read_csv(
    filename,
    on_bad_lines='skip',           # Skip bad lines
    na_values=['No data', 'NaN', '', ' ', '  ', "m", "4magnesium"],  # Treat these as NaN
    skipinitialspace=True,         # Strip spaces after delimiters
)

# Define expected columns and assert they exist
expected_columns = ['value', 'channel', 'side', 'module', 'full_label']
assert all(col in df.columns for col in expected_columns), \
    f"CSV is missing expected columns. Found: {df.columns.tolist()}, Expected: {expected_columns}"

# Keep only expected columns in correct order
df = df[expected_columns].copy()

# Add index column for plotting
df['index'] = range(len(df))

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print("=== Dataset Preview ===")
print(df.head(10))
print(f"\nTotal rows: {len(df)}")

# Define color map for modules
module_colors = {
    'BB': '#1f77b4',  # Blue
    'BA': '#ff7f0e',  # Orange
    'CA': '#2ca02c',  # Green
    'AA': '#d62728'   # Red
}

# Create the scatter plot
fig, ax = plt.subplots(figsize=(16, 8))

# Plot each module with its unique color
modules = df['module'].unique()
for module in sorted(modules):
    df_module = df[df['module'] == module]
    
    # Plot RHS data for this module
    df_rhs = df_module[df_module['side'] == 'RHS']
    ax.scatter(df_rhs['index'], df_rhs['value'], 
               c=module_colors[module], marker='o', s=120, alpha=0.7, 
               label=f'Module {module} RHS', edgecolors='black', linewidth=1.5)
    
    # Plot LHS data for this module
    df_lhs = df_module[df_module['side'] == 'LHS']
    ax.scatter(df_lhs['index'], df_lhs['value'], 
               c=module_colors[module], marker='^', s=120, alpha=0.7, 
               label=f'Module {module} LHS', edgecolors='black', linewidth=1.5)

# Customize the plot
ax.set_xlabel('Data Point Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Breakdown Voltage (V)', fontsize=12, fontweight='bold')
ax.set_title('SiPM Breakdown Voltage by Channel (Color-coded by Module)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, ncol=1)
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics text
stats_text = f"Total Points: {len(df)} | Modules: {len(modules)} | Circle=RHS, Triangle=LHS"
ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total data points: {len(df)}")
print(f"Number of modules: {len(modules)}")
print(f"Modules: {sorted(modules)}")

print(f"\nBreakdown by module:")
for module in sorted(modules):
    df_mod = df[df['module'] == module]
    rhs_count = len(df_mod[df_mod['side'] == 'RHS'])
    lhs_count = len(df_mod[df_mod['side'] == 'LHS'])
    print(f"  Module {module}: {len(df_mod)} total (RHS: {rhs_count}, LHS: {lhs_count})")

print(f"\nBreakdown by channel (across all modules):")
channels = df['channel'].unique()
for channel in sorted(channels):
    df_ch = df[df['channel'] == channel]
    print(f"  {channel}: {len(df_ch)} points")

print(f"\nValue statistics:")
print(f"  Range: {df['value'].min():.2f} to {df['value'].max():.2f}")
print(f"  Mean: {df['value'].mean():.2f}")
print(f"  Median: {df['value'].median():.2f}")
print(f"  Std Dev: {df['value'].std():.2f}")

# Identify outliers (values more than 2 std deviations from mean)
mean_val = df['value'].mean()
std_val = df['value'].std()
outliers = df[(df['value'] < mean_val - 2*std_val) | (df['value'] > mean_val + 2*std_val)]
if len(outliers) > 0:
    print(f"\nOutliers detected ({len(outliers)} points):")
    for _, row in outliers.iterrows():
        print(f"  Index {row['index']}: {row['value']:.2f} ({row['full_label']})")

# Display first few rows
print("\n=== First 10 Data Points ===")
print(df[['index', 'value', 'channel', 'side', 'module']].head(10).to_string(index=False))
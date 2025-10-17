import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




# Raw data
raw_data = """51.87421251 ch0_RHS_Module_BB 
51.783326 ch0_LHS_Module_BB 52.3869406 ch1_RHS_Module_BB 51.78537331 ch1_LHS_Module_BB 51.8975747 ch2_RHS_Module_BB 51.67767225 ch2_LHS_Module_BB 51.95217638 ch3_RHS_Module_BB 51.78332863 ch3_LHS_Module_BB 51.41044108 ch4_RHS_Module_BB 51.31823729 ch4_LHS_Module_BB 51.82818417 ch5_RHS_Module_BB 51.4905265 ch5_LHS_Module_BB 51.89313757 ch6_RHS_Module_BB 51.95092462 ch6_LHS_Module_BB 51.83161037 ch7_RHS_Module_BB 51.65570942 ch7_LHS_Module_BB
51.67786129 ch0_RHS_Module_BA 51.79866547 ch0_LHS_Module_BA 51.65539131 ch1_RHS_Module_BA 51.69749917 ch1_LHS_Module_BA 52.13177953 ch2_RHS_Module_BA 51.55819372 ch2_LHS_Module_BA 51.48400555 ch3_RHS_Module_BA 51.44514463 ch3_LHS_Module_BA 51.92831829 ch4_RHS_Module_BA 51.55922393 ch4_LHS_Module_BA 51.34235608 ch5_RHS_Module_BA 51.85951421 ch5_LHS_Module_BA 51.40337831 ch6_RHS_Module_BA 52.01275874 ch6_LHS_Module_BA 51.82412751 ch7_RHS_Module_BA 52.02758129 ch7_LHS_Module_BA
50.44184462 ch0_RHS_Module_CA 50.75496676 ch0_LHS_Module_CA 51.09632994 ch1_RHS_Module_CA 51.22559748 ch1_LHS_Module_CA 50.13767957 ch2_RHS_Module_CA 52.11943337 ch2_LHS_Module_CA 50.44958587 ch3_RHS_Module_CA 51.14508387 ch3_LHS_Module_CA 51.27346224 ch4_RHS_Module_CA 50.75592532 ch4_LHS_Module_CA 51.56941718 ch5_RHS_Module_CA 51.285531 ch5_LHS_Module_CA 51.07438198 ch6_RHS_Module_CA 51.38462391 ch6_LHS_Module_CA 51.38802818 ch7_RHS_Module_CA 50.89949182 ch7_LHS_Module_CA
51.40116407 ch0_RHS_Module_AA 50.51969901 ch0_LHS_Module_AA 28.5 ch1_RHS_Module_AA 50.86534697 ch1_LHS_Module_AA 50.85818132 ch2_RHS_Module_AA 51.37085758 ch2_LHS_Module_AA 50.65858452 ch3_RHS_Module_AA 50.69460754 ch3_LHS_Module_AA 50.478488 ch4_RHS_Module_AA 50.68910067 ch4_LHS_Module_AA 54.72302507 ch5_RHS_Module_AA 29.5 ch5_LHS_Module_AA 51.17220358 ch6_RHS_Module_AA 50.57740848 ch6_LHS_Module_AA 50.88067378 ch7_RHS_Module_AA 50.8316152 ch7_LHS_Module_AA"""

# Parse the data
data_list = []
index = 0

for line in raw_data.strip().split('\n'):
    parts = line.strip().split()
    
    # Process pairs of value and channel_info
    i = 0
    while i < len(parts):
        value = float(parts[i])
        channel_info = parts[i + 1]
        
        # Extract channel number, side, and module
        parts_info = channel_info.split('_')
        channel_num = parts_info[0]
        side = parts_info[1]
        module_name = parts_info[3]  # BB, BA, CA, or AA
        
        data_list.append({
            'index': index,
            'value': value,
            'channel': channel_num,
            'side': side,
            'module': module_name,
            'full_label': f"{channel_num}_{side}_{module_name}"
        })
        
        index += 1
        i += 2

# Create DataFrame

df = pd.DataFrame(data_list)

# Show all columns
pd.set_option('display.max_columns', None)

# Show all rows
pd.set_option('display.max_rows', None)

print(df)

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
ax.set_ylabel('Module Value', fontsize=12, fontweight='bold')
ax.set_title('Module Channel Data Scatter Plot (Color-coded by Module)', 
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
        print(f"  Index {row['index']}: {row['value']:.2f} ({row['channel']}_{row['side']}_Module_{row['module']})")

# Display first few rows
print("\n=== First 10 Data Points ===")
print(df[['index', 'value', 'channel', 'side', 'module']].head(10).to_string(index=False))
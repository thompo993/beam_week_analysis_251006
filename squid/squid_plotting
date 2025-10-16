import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MPMS3Plotter:
    """
    A class to read and plot MPMS3 .dat files from Quantum Design magnetometers.
    """
    
    def __init__(self, file_path):
        """
        Initialize the plotter with a file path.
        
        Parameters:
        -----------
        file_path : str
            Path to the MPMS3 .dat file
        """
        self.file_path = file_path
        self.header_info = {}
        self.column_names = []
        self.data = None
        
    def read_file(self):
        """
        Read the MPMS3 .dat file, extracting header info and data.
        """
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Find where [Data] section starts
        data_start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == '[Data]':
                data_start_idx = i
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find [Data] section in file")
        
        # Extract header information (before [Data])
        for line in lines[:data_start_idx]:
            if line.startswith('INFO,'):
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    key = parts[-1]
                    value = ','.join(parts[1:-1])
                    self.header_info[key] = value
        
        # Get column names (first line after [Data])
        column_line = lines[data_start_idx + 1].strip()
        self.column_names = [col.strip() for col in column_line.split(',')]
        
        # Read data (skip the header line after column names)
        data_lines = lines[data_start_idx + 2:]
        
        # Parse data into DataFrame
        data_rows = []
        for line in data_lines:
            if line.strip():  # Skip empty lines
                values = line.strip().split(',')
                data_rows.append(values)
        
        self.data = pd.DataFrame(data_rows, columns=self.column_names)
        
        # Convert numeric columns to float
        for col in self.data.columns:
            if col != 'Comment':  # Skip comment column
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                except:
                    pass
        
        return self
    
    def print_header_info(self):
        """
        Print the header information extracted from the file.
        """
        print("=" * 60)
        print("MPMS3 DATA FILE HEADER INFORMATION")
        print("=" * 60)
        for key, value in self.header_info.items():
            print(f"{key}: {value}")
        print("=" * 60)
    
    def print_columns(self):
        """
        Print available column names for plotting.
        """
        print("\nAvailable columns:")
        print("-" * 60)
        for i, col in enumerate(self.column_names, 1):
            print(f"{i}. {col}")
        print("-" * 60)
    
    def plot(self, x_column, y_column, title=None, xlabel=None, ylabel=None, 
             figsize=(10, 6), marker='o', linestyle='-', color='red', 
             markersize=4, grid=True, show_error=False, error_column=None,
             mass_norm=False, mass_grams=None):
        """
        Create a plot of the data.
        
        Parameters:
        -----------
        x_column : str
            Name of the column to use for x-axis
        y_column : str
            Name of the column to use for y-axis
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label (defaults to column name)
        ylabel : str, optional
            Y-axis label (defaults to column name)
        figsize : tuple, optional
            Figure size (width, height)
        marker : str, optional
            Marker style
        linestyle : str, optional
            Line style
        color : str, optional
            Color of the plot
        markersize : int, optional
            Size of markers
        grid : bool, optional
            Whether to show grid
        show_error : bool, optional
            Whether to show error bars
        error_column : str, optional
            Column name for error values
        mass_norm : bool, optional
            Whether to normalize by mass (default: False)
        mass_grams : float, optional
            Sample mass in grams for normalization
        """
        if self.data is None:
            raise ValueError("No data loaded. Call read_file() first.")
        
        # Validate mass normalization parameters
        if mass_norm and mass_grams is None:
            raise ValueError("mass_grams must be provided when mass_norm=True")
        if mass_norm and mass_grams <= 0:
            raise ValueError("mass_grams must be a positive number")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Get data
        x = self.data[x_column]
        y = self.data[y_column].copy()
        
        # Apply mass normalization if requested
        if mass_norm:
            y = y / mass_grams
            if ylabel is None:
                # Modify default ylabel for mass normalization
                if 'emu' in y_column.lower():
                    ylabel = y_column.replace('emu', 'emu/g').replace('(emu)', '(emu/g)')
                else:
                    ylabel = f"{y_column} (normalized by mass)"
        
        # Plot with or without error bars
        if show_error and error_column:
            yerr = self.data[error_column].copy()
            if mass_norm:
                yerr = yerr / mass_grams
            plt.errorbar(x, y, yerr=yerr, marker=marker, linestyle=linestyle, 
                        color=color, markersize=markersize, capsize=3, 
                        markerfacecolor=color, markeredgecolor='black', 
                        markeredgewidth=0.5)
        else:
            plt.plot(x, y, marker=marker, linestyle=linestyle, color=color, 
                    markersize=markersize, markerfacecolor=color, 
                    markeredgecolor='black', markeredgewidth=0.5)
        
        # Set labels and title
        plt.xlabel(xlabel if xlabel else x_column, fontsize=12)
        plt.ylabel(ylabel if ylabel else y_column, fontsize=12)
        
        # Modify title if mass normalized
        if title:
            plot_title = title
        else:
            plot_title = f"{y_column} vs {x_column}"
            if mass_norm:
                plot_title += f" (normalized by {mass_grams} g)"
        plt.title(plot_title, fontsize=14)
        
        # Add grid
        if grid:
            plt.grid(True, alpha=0.3, linestyle='--')
        
        # Format plot
        plt.tight_layout()
        
        return plt
    
    def plot_moment_vs_field(self, show_error=False, mass_norm=False, mass_grams=None, **kwargs):
        """
        Convenience method to plot Moment vs Magnetic Field (common plot).
        
        Parameters:
        -----------
        show_error : bool, optional
            Whether to display error bars (default: False)
        mass_norm : bool, optional
            Whether to normalize by mass (default: False)
        mass_grams : float, optional
            Sample mass in grams for normalization
        """
        return self.plot(
            x_column='Magnetic Field (Oe)',
            y_column='Moment (emu)',
            error_column='M. Std. Err. (emu)' if show_error else None,
            show_error=show_error,
            mass_norm=mass_norm,
            mass_grams=mass_grams,
            xlabel='Magnetic Field (Oe)',
            ylabel='Moment (emu/g)' if mass_norm else 'Moment (emu)',
            title='Magnetic Moment vs Field',
            **kwargs
        )


def plot_overlay(file_configs, x_column='Magnetic Field (Oe)', y_column='Moment (emu)',
                 title='Overlay Plot', xlabel=None, ylabel=None, figsize=(10, 6),
                 grid=True, show_error=False, error_column='M. Std. Err. (emu)'):
    """
    Create an overlay plot from multiple MPMS3 files.
    
    Parameters:
    -----------
    file_configs : list of dict
        List of configuration dictionaries for each file. Each dict should contain:
        - 'path': str (required) - Path to the file
        - 'label': str (optional) - Label for the legend
        - 'color': str (optional) - Line/marker color
        - 'marker': str (optional) - Marker style
        - 'mass_grams': float (optional) - Mass for normalization
        - 'mass_norm': bool (optional) - Whether to normalize by mass
    x_column : str
        Column name for x-axis
    y_column : str
        Column name for y-axis
    title : str
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    figsize : tuple
        Figure size
    grid : bool
        Whether to show grid
    show_error : bool
        Whether to show error bars
    error_column : str
        Column name for error values
    """
    plt.figure(figsize=figsize)
    
    # Default colors if not specified
    default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    for i, config in enumerate(file_configs):
        # Extract configuration
        file_path = config['path']
        label = config.get('label', f"File {i+1}")
        color = config.get('color', default_colors[i % len(default_colors)])
        marker = config.get('marker', default_markers[i % len(default_markers)])
        mass_norm = config.get('mass_norm', False)
        mass_grams = config.get('mass_grams', None)
        markersize = config.get('markersize', 4)
        linestyle = config.get('linestyle', '-')
        
        # Read file
        plotter = MPMS3Plotter(file_path)
        plotter.read_file()
        
        # Get data
        x = plotter.data[x_column]
        y = plotter.data[y_column].copy()
        
        # Apply mass normalization if requested
        if mass_norm:
            if mass_grams is None:
                raise ValueError(f"mass_grams must be provided for {label} when mass_norm=True")
            y = y / mass_grams
        
        # Plot with or without error bars
        if show_error and error_column:
            yerr = plotter.data[error_column].copy()
            if mass_norm:
                yerr = yerr / mass_grams
            plt.errorbar(x, y, yerr=yerr, marker=marker, linestyle=linestyle,
                        color=color, markersize=markersize, capsize=3,
                        markerfacecolor=color, markeredgecolor='black',
                        markeredgewidth=0.5, label=label, alpha=0.7)
        else:
            plt.plot(x, y, marker=marker, linestyle=linestyle, color=color,
                    markersize=markersize, markerfacecolor=color,
                    markeredgecolor='black', markeredgewidth=0.5, label=label, alpha=0.7)
    
    # Set labels and title
    plt.xlabel(xlabel if xlabel else x_column, fontsize=12)
    plt.ylabel(ylabel if ylabel else y_column, fontsize=12)
    plt.title(title, fontsize=14)
    
    # Add legend
    plt.legend(fontsize=10, loc='best')
    
    # Add grid
    if grid:
        plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format plot
    plt.tight_layout()
    
    return plt


# Example usage
if __name__ == "__main__":
    
    # Define your file paths and configurations
    file_configs = [
        # {
        #     'path': r"C:\Users\fzy12567\OneDrive - Science and Technology Facilities Council\Ben Thompson 2025-2026 Onedrive\SQUID\3M 425 DWB - Magnetic Properties Test 251015\data\alu_tape_3M_425_DWB-15102025.dat",
        #     'label': 'Initial Square Aluminum Tape Sample',
        #     'color': 'red',
        #     'marker': 'o',
        #     'mass_norm': True,
        #     'mass_grams': 0.00645,
        #     'markersize': 4
        # },
        # {
        #     'path': r"C:\Users\fzy12567\OneDrive - Science and Technology Facilities Council\Ben Thompson 2025-2026 Onedrive\SQUID\3M 425 DWB - Magnetic Properties Test 251015\data\alu_tape_3M_425_DWB-15102025-2stepped.dat",
        #     'label':"Initial Square Shape Stepped Aluminum Tape Sample",
        #     'color': 'red',
        #     'marker': 's',
        #     'mass_norm': True,
        #     'mass_grams': 0.00645,
        #     'markersize': 4
        # },
        {
            'path': r"C:\Users\fzy12567\OneDrive - Science and Technology Facilities Council\Ben Thompson 2025-2026 Onedrive\SQUID\3M 425 DWB - Magnetic Properties Test 251015\data\alu_tape_3M_425_DWB-15102025-2stepped_A_ball_00001.dat",
            'label':"Ball with Adhesive",
            'color': 'yellow',
            'marker': 's',
            'mass_norm': True,
            'mass_grams': 0.03204,
            'markersize': 4
        },
        # {
        #     'path': r"C:\Users\fzy12567\OneDrive - Science and Technology Facilities Council\Ben Thompson 2025-2026 Onedrive\SQUID\3M 425 DWB - Magnetic Properties Test 251015\data\alu_tape_3M_425_DWB-15102025-2stepped_FlatPlate_with_Adhesive.dat",
        #     'label':"square shape with adhesive",
        #     'color': 'red',
        #     'marker': 's',
        #     'mass_norm': True,
        #     'mass_grams': 0.04136,
        #     'markersize': 4
        # },
         {
            'path': r"C:\Users\fzy12567\OneDrive - Science and Technology Facilities Council\Ben Thompson 2025-2026 Onedrive\SQUID\3M 425 DWB - Magnetic Properties Test 251015\data\alu_tape_3M_425_DWB-15102025-2stepped_NA_ball.dat",
            'label':"Ball no Adhesive",
            'color': 'purple',
            'marker': 's',
            'mass_norm': True,
            'mass_grams': 0.04309,
            'markersize': 4
        },
    ]
    
    try:
        # Plot individual files
        for i, config in enumerate(file_configs):
            print(f"\n{'='*60}")
            print(f"Processing File {i+1}: {config.get('label', 'Unnamed')}")
            print(f"{'='*60}")
            
            plotter = MPMS3Plotter(config['path'])
            plotter.read_file()
            plotter.print_header_info()
            
            # Create individual plot
            plotter.plot_moment_vs_field(
                show_error=config.get('show_error', False),
                mass_norm=config.get('mass_norm', False),
                mass_grams=config.get('mass_grams', None),
                marker=config.get('marker', 'o'),
                color=config.get('color', 'red'),
                markersize=config.get('markersize', 4)
            )
            plt.show()
        
        # Create overlay plot
        print(f"\n{'='*60}")
        print("Creating Overlay Plot")
        print(f"{'='*60}")
        
        plot_overlay(
            file_configs=file_configs,
            x_column='Magnetic Field (Oe)',
            y_column='Moment (emu)',
            title='Magnetic Moment vs Field - Overlay',
            xlabel='Magnetic Field (Oe)',
            ylabel='Moment (emu/g)',  # Adjust based on whether you're normalizing
            figsize=(10, 6),
            show_error=False  # Set to True to show error bars on overlay
        )
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in 'file_configs'.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
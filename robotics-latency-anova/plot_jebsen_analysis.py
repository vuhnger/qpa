import matplotlib.pyplot as plt
import re
import numpy as np

def parse_jebsen_analysis(file_path):
    """Parse the Jebsen Taylor analysis file and extract timing data."""
    
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all participant summary sections
    summary_pattern = r'=== PARTICIPANT (\d+) SUMMARY ===\n(.*?)(?=================================================================================|$)'
    
    for match in re.finditer(summary_pattern, content, re.DOTALL):
        participant_id = int(match.group(1))
        summary_text = match.group(2)
        
        # Parse each latency line in the summary
        # Format: P  1 | 50ms | 1098171.083s | 1098340.603s |   169.5s
        line_pattern = r'P\s+\d+\s+\|\s+(\d+)ms\s+\|\s+[\d.]+s\s+\|\s+[\d.]+s\s+\|\s+([\d.]+)s'
        
        for line_match in re.finditer(line_pattern, summary_text):
            latency = int(line_match.group(1))
            duration_seconds = float(line_match.group(2))
            
            data.append({
                'participant': participant_id,
                'latency': latency,
                'duration_seconds': duration_seconds
            })
    
    return data

def create_latency_time_plot(data):
    """Create a scatter plot of latency vs. completion time."""
    
    # Separate data by latency condition
    latencies = [0, 50, 100, 150, 200]
    latency_data = {lat: [] for lat in latencies}
    
    for point in data:
        if point['latency'] in latencies:
            latency_data[point['latency']].append(point['duration_seconds'])
    
    # Create the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for each latency condition (Star Wars lightsaber colors)
    colors = ['#00BFFF', '#FFA500', '#32CD32', '#DC143C', '#9370DB']
    
    # Create scatter plot for each latency and calculate means
    mean_times = []
    mean_latencies = []
    
    for i, latency in enumerate(latencies):
        times = latency_data[latency]
        if times:  # Only plot if we have data
            # Add some jitter to x-axis for better visibility
            x_positions = [latency + np.random.normal(0, 2) for _ in times]
            ax.scatter(x_positions, times, 
                      color=colors[i], 
                      alpha=0.7, 
                      s=100, 
                      edgecolors='white', 
                      linewidth=1,
                      label=f'{latency}ms (n={len(times)})')
            
            # Calculate mean and add to lists for line plot
            mean_time = np.mean(times)
            mean_times.append(mean_time)
            mean_latencies.append(latency)
            
            # Print the mean value
            print(f"{latency}ms latency: Mean completion time = {mean_time:.1f} seconds")
            
            # Plot mean point with larger size and different style
            ax.scatter(latency, mean_time, 
                      color=colors[i], 
                      s=200, 
                      edgecolors='white', 
                      linewidth=3,
                      marker='D',  # Diamond shape for means
                      alpha=1.0,
                      zorder=10)  # Ensure means are on top
    
    # Draw line connecting the means
    if len(mean_latencies) > 1:
        ax.plot(mean_latencies, mean_times, 
                color='white', 
                linewidth=3, 
                alpha=0.8, 
                zorder=5)
    
    # Add a dummy scatter for the mean diamond legend entry
    if mean_times:  # Only add if we have means to show
        ax.scatter([], [], 
                  color='white', 
                  s=80, 
                  edgecolors='white', 
                  linewidth=2,
                  marker='D',
                  alpha=1.0,
                  label='Mean completion time')
    
    # Customize the plot
    ax.set_xlabel('Latency Condition (milliseconds)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Task Completion Time (seconds)', fontsize=24, fontweight='bold')
    ax.set_title('Participants time used to complete JTHFT for each latency', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(latencies)
    ax.set_xticklabels([f'{lat}ms' for lat in latencies])
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Set dark theme with white grid lines
    ax.set_facecolor('#1a1a1a')  # Dark gray background
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', color='white', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Add summary statistics text
    summary_text = f"Total participants analyzed: {len(set(point['participant'] for point in data))}\n"
    summary_text += f"Total data points: {len(data)}"
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=16, 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    # Parse the analysis file
    print("Parsing Jebsen Taylor analysis file...")
    data = parse_jebsen_analysis('all_participants_jebsen_analysis.txt')
    
    print(f"Found {len(data)} data points from {len(set(point['participant'] for point in data))} participants")
    
    # Create the plot
    print("Creating latency vs. completion time plot...")
    fig, ax = create_latency_time_plot(data)
    
    # Save the plot
    plt.savefig('jebsen_taylor_latency_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.savefig('jebsen_taylor_latency_analysis.pdf', bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    
    # Show statistics
    print("\\n=== SUMMARY STATISTICS ===")
    latencies = [0, 50, 100, 150, 200]
    for latency in latencies:
        latency_times = [point['duration_seconds'] for point in data if point['latency'] == latency]
        if latency_times:
            mean_time = np.mean(latency_times)
            std_time = np.std(latency_times)
            print(f"{latency:3d}ms: {len(latency_times):2d} participants, "
                  f"Mean: {mean_time:.1f}±{std_time:.1f} seconds")
        else:
            print(f"{latency:3d}ms: No data")
    
    plt.show()
    print("\\nPlot saved as 'jebsen_taylor_latency_analysis.png' and '.pdf'")
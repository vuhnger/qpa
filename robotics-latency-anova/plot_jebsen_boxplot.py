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

def create_latency_boxplot(data):
    """Create a box plot of latency vs. completion time."""
    
    # Separate data by latency condition
    latencies = [0, 50, 100, 150, 200]
    latency_data = {lat: [] for lat in latencies}
    
    for point in data:
        if point['latency'] in latencies:
            latency_data[point['latency']].append(point['duration_seconds'])
    
    # Prepare data for boxplot (only include conditions with data)
    box_data = []
    box_labels = []
    box_positions = []
    
    for latency in latencies:
        times = latency_data[latency]
        if times:  # Only include if we have data
            box_data.append(times)
            box_labels.append(f'{latency}ms')
            box_positions.append(latency)
    
    # Create the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for each latency condition (Star Wars lightsaber colors)
    colors = ['#00BFFF', '#FFA500', '#32CD32', '#DC143C', '#9370DB']
    
    # Create custom boxplot
    bp = ax.boxplot(box_data, 
                    positions=box_positions,
                    widths=15,  # Width of boxes
                    patch_artist=True,  # Fill boxes with colors
                    showmeans=True,  # Show mean markers
                    meanline=True,  # Show mean as line instead of point
                    whis=[0, 100],  # Set whiskers to show min and max (0th and 100th percentile)
                    )
    
    # Customize box colors and styles
    for i, (patch, latency) in enumerate(zip(bp['boxes'], box_positions)):
        # Find the color index for this latency
        color_idx = latencies.index(latency)
        patch.set_facecolor(colors[color_idx])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    
    # Customize whiskers and caps
    for element in ['whiskers', 'caps']:
        for item in bp[element]:
            item.set_color('white')
            item.set_linewidth(2)
    
    # Hide median lines by making them transparent
    for item in bp['medians']:
        item.set_alpha(0)
    
    # Customize mean lines
    for item in bp['means']:
        item.set_color('yellow')
        item.set_linewidth(3)
        item.set_alpha(0.9)
    
    # Print statistics for each condition
    print("\n=== BOXPLOT STATISTICS ===")
    for i, (latency, times) in enumerate(zip(box_positions, box_data)):
        mean_time = np.mean(times)
        median_time = np.median(times)
        min_time = np.min(times)
        max_time = np.max(times)
        q1_time = np.percentile(times, 25)
        q3_time = np.percentile(times, 75)
        
        print(f"{latency}ms latency (n={len(times)}):")
        print(f"  Mean: {mean_time:.1f}s, Median: {median_time:.1f}s")
        print(f"  Range: {min_time:.1f}s - {max_time:.1f}s")
        print(f"  Q1-Q3: {q1_time:.1f}s - {q3_time:.1f}s")
        print()
    
    # Customize the plot
    ax.set_xlabel('Latency Condition (milliseconds)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Task Completion Time (seconds)', fontsize=24, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(box_positions)
    ax.set_xticklabels([f'{lat}ms' for lat in box_positions], fontsize=20)
    
    # Set y-axis tick labels size
    ax.tick_params(axis='y', labelsize=20)
    
    # Set dark theme with white grid lines
    ax.set_facecolor('#1a1a1a')  # Dark gray background
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', color='white', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='white', alpha=0.7, edgecolor='white', linewidth=2, label='Interquartile Range (Q1-Q3)'),
        Line2D([0], [0], color='yellow', linewidth=3, label='Mean'),
        Line2D([0], [0], color='white', linewidth=2, label='Min/Max (Whiskers)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18)
    
    # Add summary statistics text
    total_participants = len(set(point['participant'] for point in data))
    total_datapoints = len(data)
    summary_text = f"Total participants analyzed: {total_participants}\n"
    summary_text += f"Total data points: {total_datapoints}\n"
    
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
    
    # Create the boxplot
    print("Creating latency vs. completion time boxplot...")
    fig, ax = create_latency_boxplot(data)
    
    # Save the plot
    plt.savefig('jebsen_taylor_latency_boxplot.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.savefig('jebsen_taylor_latency_boxplot.pdf', bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    
    plt.show()
    print("\nBoxplot saved as 'jebsen_taylor_latency_boxplot.png' and '.pdf'")
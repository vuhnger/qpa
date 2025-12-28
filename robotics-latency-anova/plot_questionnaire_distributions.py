#!/usr/bin/env python3
"""
Script to create distribution plots for questionnaire responses.
Creates two histograms showing the frequency of each response value (1-5) for both questionnaire questions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    """Load the questionnaire data from CSV file."""
    try:
        df = pd.read_csv(filepath, delimiter=';')
        print(f"Loaded data with {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def plot_response_distribution(data, question_col, title, ax):
    """Plot the distribution of responses for a single question grouped by latency."""
    # Filter out missing values and unknown latencies
    clean_data = data[data['Latency'] != 'Unknown'].dropna(subset=[question_col])
    
    # Define latency conditions and colors (Star Wars lightsaber spectrum)
    latencies = ['0ms', '50ms', '100ms', '150ms', '200ms']
    colors = ['#00BFFF', '#FFA500', '#32CD32', '#DC143C', '#9370DB']  # Star Wars lightsaber colors
    
    # Prepare data structure for grouped bars
    response_values = range(1, 6)
    bar_width = 0.15
    x_positions = np.arange(len(response_values))
    
    # Create grouped bars
    bars_by_latency = []
    for i, latency in enumerate(latencies):
        latency_data = clean_data[clean_data['Latency'] == latency][question_col]
        
        # Count responses for each value (1-5)
        counts = []
        for response_val in response_values:
            count = (latency_data == response_val).sum()
            counts.append(count)
        
        # Position bars for this latency condition
        bar_positions = x_positions + (i - 2) * bar_width  # Center around x_positions
        bars = ax.bar(bar_positions, counts, bar_width, 
                     label=latency, color=colors[i], alpha=0.85, 
                     edgecolor='white', linewidth=0.8)
        bars_by_latency.append((bars, counts))
    
    # Customize the plot
    ax.set_title(title, fontsize=22, fontweight='bold', pad=25)
    ax.set_xlabel('Response Value (Likert Scale)', fontsize=18)
    ax.set_ylabel('Number of Participants', fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(response_values)
    ax.set_xlim(-0.5, len(response_values) - 0.5)
    
    # Add legend
    ax.legend(title='Latency', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=18)
    
    # Add value labels on top of bars (only for non-zero values)
    for bars, counts in bars_by_latency:
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                       f'{count}', ha='center', va='bottom', fontsize=14, color='white', weight='bold')
    
    # Set dark theme with white grid lines
    ax.set_facecolor('#1a1a1a')  # Dark gray background
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', color='white', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set text colors to white for dark theme
    ax.tick_params(colors='white', labelsize=16)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    
    # Display statistics with dark theme styling
    total_responses = len(clean_data)
    mean_response = clean_data[question_col].mean()
    ax.text(0.02, 0.98, f'Total responses: {total_responses}\nMean: {mean_response:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.9, edgecolor='white'),
            fontsize=14, color='white')

def create_distribution_plots(filepath):
    """Create and save distribution plots for both questionnaire questions."""
    # Load data
    df = load_data(filepath)
    if df is None:
        return
    
    # Define question columns and titles
    questions = {
        'I felt like I was controlling the movement of the robot': 'I felt like I was controlling the movement of the robot',
        'It felt like the robot was part of my body': 'It felt like the robot was part of my body'
    }
    
    # Create figure with subplots and dark theme
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), facecolor='#1a1a1a')
    
    # Create plots for each question
    for i, (question_col, title) in enumerate(questions.items()):
        if question_col in df.columns:
            plot_response_distribution(df, question_col, title, axes[i])
        else:
            print(f"Warning: Column '{question_col}' not found in data")
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.85)  # Make room for main title and legend
    
    # Save the plot
    output_filename = 'questionnaire_distributions_by_latency.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics by latency
    print("\n=== Summary Statistics by Latency ===")
    clean_data = df[df['Latency'] != 'Unknown']
    latencies = ['0ms', '50ms', '100ms', '150ms', '200ms']
    
    for question_col in questions.keys():
        if question_col in df.columns:
            print(f"\n{question_col}:")
            for latency in latencies:
                latency_data = clean_data[clean_data['Latency'] == latency][question_col].dropna()
                if len(latency_data) > 0:
                    print(f"\n  {latency} latency:")
                    print(f"    Participants: {len(latency_data)}")
                    print(f"    Mean: {latency_data.mean():.2f}")
                    print(f"    Std: {latency_data.std():.2f}")
                    print(f"    Response counts:")
                    for value in range(1, 6):
                        count = (latency_data == value).sum()
                        percentage = (count / len(latency_data)) * 100 if len(latency_data) > 0 else 0
                        print(f"      {value}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    # File path to the data
    data_file = "manually_pruned_data.csv"
    
    # Create the distribution plots
    create_distribution_plots(data_file)
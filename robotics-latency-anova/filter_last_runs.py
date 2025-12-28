#!/usr/bin/env python3
"""
Filter CSV to keep only the last run (highest run number) per participant per latency condition
"""

import pandas as pd

def filter_to_last_runs(input_file, output_file):
    """
    Read CSV and filter to keep only the last run per participant per latency
    """
    # Read the CSV
    df = pd.read_csv(input_file)
    
    print(f"Original data: {len(df)} rows")
    
    # Remove rows with N/A values first
    df_clean = df[df['I felt like I was controlling the movement of the robot'] != 'N/A'].copy()
    print(f"After removing N/A rows: {len(df_clean)} rows")
    
    # Group by Participant_ID and Latency, then take the row with maximum Run_Number
    df_filtered = df_clean.loc[df_clean.groupby(['Participant_ID', 'Latency'])['Run_Number'].idxmax()]
    
    print(f"After keeping only last runs: {len(df_filtered)} rows")
    
    # Sort by participant and latency for easy reading
    latency_order = {'0ms': 0, '50ms': 1, '100ms': 2, '150ms': 3, '200ms': 4, 'Unknown': 999}
    df_filtered['latency_sort'] = df_filtered['Latency'].map(latency_order)
    df_filtered = df_filtered.sort_values(['Participant_ID', 'latency_sort'])
    df_filtered = df_filtered.drop('latency_sort', axis=1)
    
    # Save filtered data with integer formatting for questionnaire columns
    # Create a copy to modify for output
    df_output = df_filtered.copy()
    
    # Convert questionnaire columns to integers where possible (handle NaN values)
    questionnaire_cols = ['I felt like I was controlling the movement of the robot', 
                         'It felt like the robot was part of my body']
    
    for col in questionnaire_cols:
        # Convert floats to integers, but keep NaN as empty strings for CSV
        df_output[col] = df_output[col].apply(lambda x: int(x) if pd.notna(x) else '')
    
    df_output.to_csv(output_file, index=False)
    
    # Also save as Excel file
    excel_output_file = output_file.replace('.csv', '.xlsx')
    try:
        df_output.to_excel(excel_output_file, index=False)
        excel_saved = True
    except ImportError:
        print("Warning: openpyxl not installed. Install with: pip install openpyxl")
        print("Skipping Excel file creation.")
        excel_saved = False
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Participants with data: {df_filtered['Participant_ID'].nunique()}")
    print(f"Latency conditions: {sorted(df_filtered['Latency'].unique())}")
    
    print(f"\nRuns per latency condition:")
    latency_counts = df_filtered['Latency'].value_counts().sort_index()
    for latency, count in latency_counts.items():
        print(f"  {latency}: {count} participants")
    
    # Show examples of what was filtered
    print(f"\nExamples of multiple runs that were filtered:")
    original_counts = df_clean.groupby(['Participant_ID', 'Latency']).size()
    multiple_runs = original_counts[original_counts > 1]
    
    for (participant, latency), count in multiple_runs.head(10).items():
        runs = df_clean[(df_clean['Participant_ID'] == participant) & 
                       (df_clean['Latency'] == latency)]['Run_Number'].tolist()
        kept_run = max(runs)
        removed_runs = [r for r in runs if r != kept_run]
        print(f"  Participant {participant}, {latency}: kept run {kept_run}, removed runs {removed_runs}")
    
    if len(multiple_runs) > 10:
        print(f"  ... and {len(multiple_runs) - 10} more cases")

if __name__ == "__main__":
    input_file = 'all_participants_analysis.csv'
    output_file = 'filtered_participants_analysis.csv'
    
    filter_to_last_runs(input_file, output_file)
    
    print(f"\nFiltered data saved to:")
    print(f"  CSV: {output_file}")
    print("Ready for statistical analysis with one data point per participant per latency condition!")
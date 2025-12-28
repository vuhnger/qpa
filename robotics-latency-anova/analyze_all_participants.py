#!/usr/bin/env python3
"""
Multi-Participant XDF and Questionnaire Analysis Script
Processes all XDF files in datasets/ and maps to questionnaire responses
"""

import pyxdf
import numpy as np
import csv
import os
import glob
import re
from datetime import datetime

def extract_participant_number(filepath):
    """Extract participant number from filepath using robust regex"""
    # Look for patterns like sub-013, sub_013, sub013 (case-insensitive)
    m = re.search(r"sub[-_]?0*(\d+)", filepath, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    
    # Secondary fallback: check path components
    parts = filepath.replace("\\", "/").split('/')
    for p in parts:
        m2 = re.match(r"sub[-_]?0*(\d+)$", p, re.IGNORECASE)
        if m2:
            try:
                return int(m2.group(1))
            except ValueError:
                continue
    return None

def process_xdf_file(filepath):
    """Process a single XDF file and extract Jebsen-Taylor data"""
    try:
        data, header = pyxdf.load_xdf(filepath)
    except Exception as e:
        return None, f"Failed to load XDF: {str(e)}"
    
    # Find ExpMarkers and LatencyMarkers streams
    exp_stream = None
    latency_stream = None
    
    for stream in data:
        name = stream['info']['name'][0] if 'name' in stream['info'] else 'Unknown'
        if name == 'ExpMarkers':
            exp_stream = stream
        elif name == 'LatencyMarkers':
            latency_stream = stream
    
    if not exp_stream or not latency_stream:
        return None, "Missing ExpMarkers or LatencyMarkers streams"
    
    try:
        # Find Jebsen-Taylor start times (excluding practice)
        jebsen_runs = []
        for i, sample in enumerate(exp_stream['time_series']):
            sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
            if 'jebsen_taylor_start' in sample_str and 'practice' not in sample_str:
                timestamp = exp_stream['time_stamps'][i]
                jebsen_runs.append(timestamp)
        
        # Build latency timeline from LatencyMarkers
        latency_timeline = []
        for i, sample in enumerate(latency_stream['time_series']):
            sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
            timestamp = latency_stream['time_stamps'][i]
            
            if 'latency_applied|' in sample_str:
                parts = sample_str.split('|')
                if len(parts) >= 2:
                    latency = parts[1]
                    latency_timeline.append((timestamp, latency))
        
        # Determine task order (Jebsen-Taylor first or second)
        first_jebsen_time = None
        first_boxblock_time = None
        
        for i, sample in enumerate(exp_stream['time_series']):
            sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
            timestamp = exp_stream['time_stamps'][i]
            
            # Check for Jebsen-Taylor markers (excluding practice)
            if 'jebsen' in sample_str.lower() and 'practice' not in sample_str.lower():
                if first_jebsen_time is None:
                    first_jebsen_time = timestamp
            
            # Check for Box&Block markers
            if any(bb_keyword in sample_str.lower() for bb_keyword in ['boxblock', 'box_block', 'block_moved']):
                if first_boxblock_time is None:
                    first_boxblock_time = timestamp
        
        # Determine task order
        if first_jebsen_time is not None and first_boxblock_time is not None:
            jebsen_taylor_first = first_jebsen_time < first_boxblock_time
            task_order = "Jebsen-Taylor_First" if jebsen_taylor_first else "Box-Block_First"
        elif first_jebsen_time is not None:
            jebsen_taylor_first = True
            task_order = "Jebsen-Taylor_Only"
        else:
            jebsen_taylor_first = True  # Default
            task_order = "Unknown"
        
        # Map each Jebsen-Taylor run to its latency
        run_data = []
        for run_num, jebsen_start_time in enumerate(jebsen_runs, 1):
            # Find the most recent latency applied before this run
            applied_latency = "Unknown"
            for lat_time, latency in reversed(latency_timeline):
                if lat_time <= jebsen_start_time:
                    applied_latency = latency
                    break
            
            run_data.append({
                'run_number': run_num,
                'latency': applied_latency,
                'timestamp': jebsen_start_time,
                'task_order': task_order
            })
        
        return run_data, None
        
    except Exception as e:
        return None, f"Error processing streams: {str(e)}"

def load_questionnaire_data(questionnaire_file):
    """Load questionnaire data and return as dictionary keyed by participant number"""
    questionnaire_data = {}
    
    try:
        with open(questionnaire_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            headers = next(reader)  # Read headers
            
            for row in reader:
                if len(row) > 2:
                    try:
                        participant_num = int(row[2].strip())
                        questionnaire_data[participant_num] = row
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error loading questionnaire: {str(e)}")
        return {}
    
    return questionnaire_data

def get_questionnaire_responses(participant_row, task_order, latency):
    """Extract questionnaire responses for a specific latency condition"""
    if not participant_row:
        return {'delay_perception': 'N/A', 'difficulty': 'N/A', 'control_feeling': 'N/A', 'embodiment': 'N/A'}
    
    # Determine which columns to use based on task order
    if "Jebsen-Taylor_First" in task_order:
        jt_start_col = 7  # 0-indexed, columns 8-27 in 1-indexed
    else:
        jt_start_col = 27  # 0-indexed, columns 28-47 in 1-indexed
    
    # Map latency to condition number
    latency_to_condition = {
        '0ms': 1, '50ms': 2, '100ms': 3, '150ms': 4, '200ms': 5
    }
    
    condition_num = latency_to_condition.get(latency, None)
    if not condition_num:
        return {'delay_perception': 'N/A', 'difficulty': 'N/A', 'control_feeling': 'N/A', 'embodiment': 'N/A'}
    
    # Calculate column indices for this condition's responses
    base_col = jt_start_col + (condition_num - 1) * 4
    
    return {
        'delay_perception': participant_row[base_col] if base_col < len(participant_row) else 'N/A',
        'difficulty': participant_row[base_col + 1] if base_col + 1 < len(participant_row) else 'N/A',
        'control_feeling': participant_row[base_col + 2] if base_col + 2 < len(participant_row) else 'N/A',
        'embodiment': participant_row[base_col + 3] if base_col + 3 < len(participant_row) else 'N/A'
    }

def main():
    # Configuration
    datasets_dir = 'datasets'
    questionnaire_file = 'questionnaire_data-561422-2025-11-21-1620(Study on latency perception and).csv'
    csv_output_file = 'all_participants_analysis.csv'
    summary_output_file = 'analysis_summary.txt'
    
    # Find all XDF files
    xdf_files = []
    for sub_dir in glob.glob(os.path.join(datasets_dir, 'sub-*')):
        if os.path.isdir(sub_dir):
            # Look for XDF files in each subdirectory
            xdf_pattern = os.path.join(sub_dir, '*.xdf')
            files = glob.glob(xdf_pattern)
            if files:
                # Use the main XDF file (not practice files)
                main_file = None
                for f in files:
                    if 'practice' not in os.path.basename(f):
                        main_file = f
                        break
                if main_file:
                    xdf_files.append(main_file)
                else:
                    xdf_files.append(files[0])  # Fallback to first file
    
    print(f"Found {len(xdf_files)} XDF files to process")
    
    # Load questionnaire data
    questionnaire_data = load_questionnaire_data(questionnaire_file)
    print(f"Loaded questionnaire data for {len(questionnaire_data)} participants")
    
    # Process all files
    all_results = []
    processing_stats = {
        'total_files': len(xdf_files),
        'successful': 0,
        'failed': 0,
        'no_questionnaire': 0,
        'errors': []
    }
    
    with open(summary_output_file, 'w') as summary_f:
        summary_f.write("=" * 80 + "\n")
        summary_f.write("MULTI-PARTICIPANT JEBSEN-TAYLOR ANALYSIS SUMMARY\n")
        summary_f.write("=" * 80 + "\n")
        summary_f.write(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_f.write(f"Total XDF files found: {len(xdf_files)}\n")
        summary_f.write(f"Questionnaire responses available: {len(questionnaire_data)}\n\n")
        
        for i, filepath in enumerate(sorted(xdf_files)):
            participant_num = extract_participant_number(filepath)
            print(f"Processing [{i+1}/{len(xdf_files)}]: {os.path.basename(filepath)} (Participant {participant_num})")
            
            summary_f.write(f"[{i+1:2d}] Participant {participant_num:3} ({os.path.basename(filepath):<35}): ")
            
            if participant_num is None:
                error_msg = "Could not extract participant number"
                processing_stats['errors'].append(f"Participant ?: {error_msg}")
                processing_stats['failed'] += 1
                summary_f.write(f"FAILED - {error_msg}\n")
                continue
            
            # Process XDF file
            run_data, error = process_xdf_file(filepath)
            if error:
                processing_stats['errors'].append(f"Participant {participant_num}: {error}")
                processing_stats['failed'] += 1
                summary_f.write(f"FAILED - {error}\n")
                continue
            
            # Check if questionnaire data exists
            participant_row = questionnaire_data.get(participant_num)
            if not participant_row:
                processing_stats['no_questionnaire'] += 1
                summary_f.write(f"SUCCESS (XDF) - No questionnaire data\n")
            else:
                processing_stats['successful'] += 1
                summary_f.write(f"SUCCESS - {len(run_data)} runs mapped\n")
            
            # Process each run and collect data
            for run_info in run_data:
                responses = get_questionnaire_responses(
                    participant_row, 
                    run_info['task_order'], 
                    run_info['latency']
                )
                
                result = {
                    'Participant_ID': participant_num,
                    'Latency': run_info['latency'],
                    'Run_Number': run_info['run_number'],
                    'I felt like I was controlling the movement of the robot': responses['control_feeling'],
                    'It felt like the robot was part of my body': responses['embodiment']
                }
                all_results.append(result)
        
        # Write summary statistics
        summary_f.write("\n" + "=" * 80 + "\n")
        summary_f.write("PROCESSING STATISTICS\n")
        summary_f.write("=" * 80 + "\n")
        summary_f.write(f"Total files processed: {processing_stats['total_files']}\n")
        summary_f.write(f"Successful (with questionnaire): {processing_stats['successful']}\n")
        summary_f.write(f"XDF processed (no questionnaire): {processing_stats['no_questionnaire']}\n")
        summary_f.write(f"Failed to process: {processing_stats['failed']}\n")
        summary_f.write(f"Total data rows generated: {len(all_results)}\n")
        
        if processing_stats['errors']:
            summary_f.write(f"\nERRORS ENCOUNTERED ({len(processing_stats['errors'])}):\n")
            summary_f.write("-" * 50 + "\n")
            for error in processing_stats['errors']:
                summary_f.write(f"  {error}\n")
        
        # Data quality summary
        summary_f.write(f"\n" + "=" * 80 + "\n")
        summary_f.write("DATA QUALITY SUMMARY\n")
        summary_f.write("=" * 80 + "\n")
        
        participants_with_data = set(result['Participant_ID'] for result in all_results if result['I felt like I was controlling the movement of the robot'] != 'N/A')
        latency_distribution = {}
        
        for result in all_results:
            latency = result['Latency']
            latency_distribution[latency] = latency_distribution.get(latency, 0) + 1
        
        summary_f.write(f"Participants with complete data: {len(participants_with_data)}\n")
        summary_f.write(f"Total experimental runs: {len([r for r in all_results if r['Latency'] != 'Unknown'])}\n")
        
        summary_f.write(f"\nLatency condition distribution:\n")
        for latency in sorted(latency_distribution.keys()):
            summary_f.write(f"  {latency}: {latency_distribution[latency]} runs\n")
    
    # Write CSV output
    if all_results:
        with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Participant_ID', 'Latency', 'Run_Number',
                         'I felt like I was controlling the movement of the robot',
                         'It felt like the robot was part of my body']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Sort by participant ID and then by latency order for easy analysis
            latency_order = {'0ms': 0, '50ms': 1, '100ms': 2, '150ms': 3, '200ms': 4, 'Unknown': 999}
            all_results.sort(key=lambda x: (x['Participant_ID'], latency_order.get(x['Latency'], 999)))
            
            writer.writerows(all_results)
    
    print(f"\nAnalysis complete!")
    print(f"Results written to: {csv_output_file}")
    print(f"Summary written to: {summary_output_file}")
    print(f"Processed {processing_stats['successful']} participants successfully")
    print(f"Failed to process {processing_stats['failed']} files")
    if processing_stats['errors']:
        print(f"Errors: {len(processing_stats['errors'])}")

if __name__ == "__main__":
    main()
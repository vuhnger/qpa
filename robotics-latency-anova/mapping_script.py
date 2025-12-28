import pyxdf
import numpy as np
import csv
import os

# Configuration
filename = 'datasets/sub-013/sub-013_ses-_task-_run-001.xdf'
questionnaire_file = 'questionnaire_data-561422-2025-11-17-1240(Study on latency perception and).csv'
data, header = pyxdf.load_xdf(filename)

# Open output file (overwrite mode)
with open('stream_analysis_output.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("XDF FILE OVERVIEW\n")
    f.write("=" * 80 + "\n")

    f.write(f"\nFile: {filename}\n")
    f.write(f"Number of streams: {len(data)}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("STREAM DETAILS (ExpMarkers and LatencyMarkers only)\n")
    f.write("=" * 80 + "\n")

    # Filter streams to only include ExpMarkers and LatencyMarkers
    filtered_streams = []
    for stream in data:
        name = stream['info']['name'][0] if 'name' in stream['info'] else 'Unknown'
        if name in ['ExpMarkers', 'LatencyMarkers']:
            filtered_streams.append(stream)
    
    f.write(f"Found {len(filtered_streams)} marker streams out of {len(data)} total streams\n")

    for i, stream in enumerate(filtered_streams):
        f.write(f"\n[STREAM {i+1}]\n")
        f.write("-" * 40 + "\n")
        
        # Basic stream info
        name = stream['info']['name'][0] if 'name' in stream['info'] else 'Unknown'
        stream_type = stream['info']['type'][0] if 'type' in stream['info'] else 'Unknown'
        
        f.write(f"Stream Name: {name}\n")
        f.write(f"Stream Type: {stream_type}\n")
        
        # Time series data
        time_series = stream['time_series']
        time_stamps = stream['time_stamps']
        
        f.write(f"Number of samples: {len(time_series)}\n")
        f.write(f"Data shape: {np.array(time_series).shape}\n")
        
        # Time information
        if len(time_stamps) > 0:
            duration = time_stamps[-1] - time_stamps[0]
            sample_rate = len(time_stamps) / duration if duration > 0 else 0
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Approximate sample rate: {sample_rate:.2f} Hz\n")
            f.write(f"First timestamp: {time_stamps[0]:.6f}\n")
            f.write(f"Last timestamp: {time_stamps[-1]:.6f}\n")
        
        # Channel information
        if 'channel_count' in stream['info']:
            channel_count = int(stream['info']['channel_count'][0])
            f.write(f"Number of samples: {len(time_series)}\n")
        
        # Show samples (filter Jebsen-Taylor for ExpMarkers)
        if len(time_series) > 0:
            if name == 'ExpMarkers':
                # Filter for Jebsen-Taylor related markers only
                jebsen_samples = []
                jebsen_timestamps = []
                
                for j in range(len(time_series)):
                    sample = time_series[j]
                    timestamp = time_stamps[j]
                    
                    # Convert sample to string for checking
                    sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
                    
                    # Include Jebsen-Taylor related markers and exclude boxblock markers
                    jebsen_keywords = ['jebsen', 'taylor', 'practice_jebsen', 'JEBSEN_TAYLOR']
                    boxblock_keywords = ['boxblock', 'block_moved', 'BOX_BLOCK']
                    
                    # Check if it's a Jebsen-Taylor marker
                    is_jebsen = any(keyword.lower() in sample_str.lower() for keyword in jebsen_keywords)
                    # Check if it's a boxblock marker
                    is_boxblock = any(keyword.lower() in sample_str.lower() for keyword in boxblock_keywords)
                    
                    # Include general markers (countdown, trial_start, etc.) and Jebsen-Taylor specific markers
                    # Exclude boxblock specific markers
                    if not is_boxblock and (is_jebsen or any(keyword in sample_str.lower() for keyword in 
                        ['countdown', 'trial_start', 'trial_end', 'labrecorder', 'task_accept', 'session_end', 'practice_start', 'practice_end'])):
                        jebsen_samples.append(sample)
                        jebsen_timestamps.append(timestamp)
                
                f.write(f"\nJebsen-Taylor related samples ({len(jebsen_samples)} out of {len(time_series)} total):\n")
                for j, (sample, timestamp) in enumerate(zip(jebsen_samples, jebsen_timestamps)):
                    f.write(f"  [{j}] @{timestamp:.6f}s: {sample}\n")
            else:
                # For LatencyMarkers, show all samples
                f.write(f"\nAll {len(time_series)} samples:\n")
                for j in range(len(time_series)):
                    sample = time_series[j]
                    timestamp = time_stamps[j]
                    if isinstance(sample, (list, np.ndarray)) and len(sample) > 5:
                        f.write(f"  [{j}] @{timestamp:.6f}s: [{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}, ..., {sample[-1]:.3f}] (shape: {len(sample)})\n")
                    else:
                        f.write(f"  [{j}] @{timestamp:.6f}s: {sample}\n")
        
        # Additional metadata
        if 'desc' in stream['info']:
            f.write(f"\nStream description available: Yes\n")
        
        f.write("-" * 40 + "\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("JEBSEN-TAYLOR LATENCY MAPPING\n")
    f.write("=" * 80 + "\n")
    
    # Extract Jebsen-Taylor runs and latency mapping
    exp_stream = None
    latency_stream = None
    
    for stream in filtered_streams:
        name = stream['info']['name'][0] if 'name' in stream['info'] else 'Unknown'
        if name == 'ExpMarkers':
            exp_stream = stream
        elif name == 'LatencyMarkers':
            latency_stream = stream
    
    if exp_stream and latency_stream:
        # Find Jebsen-Taylor start times (excluding practice)
        jebsen_runs = []
        for i, sample in enumerate(exp_stream['time_series']):
            sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
            if 'jebsen_taylor_start' in sample_str and 'practice' not in sample_str:
                timestamp = exp_stream['time_stamps'][i]
                jebsen_runs.append(timestamp)
        
        # Build latency timeline from LatencyMarkers
        latency_timeline = []
        current_latency = None
        
        for i, sample in enumerate(latency_stream['time_series']):
            sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
            timestamp = latency_stream['time_stamps'][i]
            
            if 'latency_applied|' in sample_str:
                # Extract latency value (e.g., "latency_applied|150ms|ifb0" -> "150ms")
                parts = sample_str.split('|')
                if len(parts) >= 2:
                    current_latency = parts[1]
                    latency_timeline.append((timestamp, current_latency))
        
        # Map each Jebsen-Taylor run to its latency
        f.write("\nJebsen-Taylor Run to Latency Mapping:\n")
        f.write("-" * 40 + "\n")
        
        for run_num, jebsen_start_time in enumerate(jebsen_runs, 1):
            # Find the most recent latency applied before this run
            applied_latency = "Unknown"
            for lat_time, latency in reversed(latency_timeline):
                if lat_time <= jebsen_start_time:
                    applied_latency = latency
                    break
            
            f.write(f"Run {run_num}: {applied_latency} (started @{jebsen_start_time:.6f}s)\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"Total Jebsen-Taylor runs found: {len(jebsen_runs)}\n")
        f.write(f"Total latency changes: {len(latency_timeline)}\n")
        
        # QUESTIONNAIRE INTEGRATION
        f.write("\n" + "=" * 80 + "\n")
        f.write("QUESTIONNAIRE RESPONSE INTEGRATION\n")
        f.write("=" * 80 + "\n")
        
        # Extract participant number from filename (robustly)
        import re
        participant_num = None
        # Look for patterns like sub-013, sub_013, sub013 (case-insensitive)
        m = re.search(r"sub[-_]?0*(\d+)", filename, re.IGNORECASE)
        if m:
            try:
                participant_num = int(m.group(1))
            except ValueError:
                participant_num = None
        else:
            # As a secondary fallback, try to find any 'sub' directory in the path
            # Example: datasets/sub-013/...
            parts = filename.replace("\\", "/").split('/')
            for p in parts:
                m2 = re.match(r"sub[-_]?0*(\d+)$", p, re.IGNORECASE)
                if m2:
                    try:
                        participant_num = int(m2.group(1))
                        break
                    except ValueError:
                        continue

        if participant_num is None:
            f.write("Could not extract participant number from filename\n")
        
        if participant_num and os.path.exists(questionnaire_file):
            f.write(f"Participant: {participant_num}\n")
            f.write(f"Questionnaire file: {questionnaire_file}\n\n")
            
            try:
                # Read questionnaire data
                with open(questionnaire_file, 'r', encoding='utf-8') as csvfile:
                    # Use semicolon as delimiter based on the CSV format
                    reader = csv.reader(csvfile, delimiter=';')
                    headers = next(reader)  # Read headers
                    
                    # Find the row for this participant
                    participant_row = None
                    for row in reader:
                        if len(row) > 2 and row[2].strip() == str(participant_num):
                            participant_row = row
                            break
                
                if participant_row:
                    f.write(f"Found questionnaire data for participant {participant_num}\n")
                    
                    # Determine task order by analyzing ExpMarkers chronologically
                    first_jebsen_time = None
                    first_boxblock_time = None
                    
                    # Find the first occurrence of each task type
                    for i, sample in enumerate(exp_stream['time_series']):
                        sample_str = str(sample[0]) if isinstance(sample, list) and len(sample) > 0 else str(sample)
                        timestamp = exp_stream['time_stamps'][i]
                        
                        # Check for Jebsen-Taylor task markers (excluding practice)
                        if 'jebsen' in sample_str.lower() and 'practice' not in sample_str.lower():
                            if first_jebsen_time is None:
                                first_jebsen_time = timestamp
                        
                        # Check for Box&Block task markers
                        if any(bb_keyword in sample_str.lower() for bb_keyword in ['boxblock', 'box_block', 'block_moved']):
                            if first_boxblock_time is None:
                                first_boxblock_time = timestamp
                    
                    # Determine which task came first
                    if first_jebsen_time is not None and first_boxblock_time is not None:
                        if first_jebsen_time < first_boxblock_time:
                            jebsen_taylor_first = True
                            task_order_info = f"Jebsen-Taylor first (@{first_jebsen_time:.2f}s), Box&Block second (@{first_boxblock_time:.2f}s)"
                        else:
                            jebsen_taylor_first = False
                            task_order_info = f"Box&Block first (@{first_boxblock_time:.2f}s), Jebsen-Taylor second (@{first_jebsen_time:.2f}s)"
                    elif first_jebsen_time is not None:
                        jebsen_taylor_first = True
                        task_order_info = f"Only Jebsen-Taylor found (@{first_jebsen_time:.2f}s)"
                    elif first_boxblock_time is not None:
                        jebsen_taylor_first = False  # Box&Block must be first
                        task_order_info = f"Only Box&Block found (@{first_boxblock_time:.2f}s), assuming Jebsen-Taylor is second"
                    else:
                        jebsen_taylor_first = True  # Default fallback
                        task_order_info = "Could not detect task order from markers, defaulting to Jebsen-Taylor first"
                    
                    # The questionnaire has 40 response columns (8-47)
                    # Columns 8-27: First task (20 responses = 5 conditions × 4 questions)
                    # Columns 28-47: Second task (20 responses = 5 conditions × 4 questions)
                    
                    f.write(f"\nTask order detection: {task_order_info}\n")
                    f.write("\nQuestionnaire Response Structure:\n")
                    f.write("- Columns 8-27: First task responses (5 conditions × 4 questions)\n")
                    f.write("- Columns 28-47: Second task responses (5 conditions × 4 questions)\n")
                    f.write("- 4 questions per condition: Delay perception, Difficulty, Control feeling, Embodiment\n")
                    
                    if jebsen_taylor_first:
                        jt_start_col = 7  # 0-indexed, so column 8 in 1-indexed
                        task_label = "First Task (assumed Jebsen-Taylor)"
                    else:
                        jt_start_col = 27  # 0-indexed, so column 28 in 1-indexed  
                        task_label = "Second Task (assumed Jebsen-Taylor)"
                    
                    f.write(f"\nUsing {task_label} responses for Jebsen-Taylor analysis\n")
                    f.write("\n" + "-" * 60 + "\n")
                    f.write("RUN → LATENCY → QUESTIONNAIRE RESPONSES\n")
                    f.write("-" * 60 + "\n")
                    
                    # Map each run to its responses and collect data
                    latency_to_condition = {
                        '0ms': 1, '50ms': 2, '100ms': 3, '150ms': 4, '200ms': 5
                    }
                    
                    # Collect run data for sorting by latency
                    run_data = []
                    for run_num, jebsen_start_time in enumerate(jebsen_runs, 1):
                        # Find the latency for this run
                        applied_latency = "Unknown"
                        for lat_time, latency in reversed(latency_timeline):
                            if lat_time <= jebsen_start_time:
                                applied_latency = latency
                                break
                        
                        run_data.append((run_num, applied_latency, jebsen_start_time))
                    
                    # Sort by latency value (0ms, 50ms, 100ms, 150ms, 200ms)
                    latency_order = ['0ms', '50ms', '100ms', '150ms', '200ms']
                    run_data.sort(key=lambda x: latency_order.index(x[1]) if x[1] in latency_order else 999)
                    
                    # Output in latency order
                    for run_num, applied_latency, jebsen_start_time in run_data:
                        # Map latency to condition number
                        condition_num = latency_to_condition.get(applied_latency, None)
                        
                        if condition_num:
                            # Calculate column indices for this condition's responses
                            # Each condition has 4 questions, so condition N uses columns:
                            # jt_start_col + (condition_num-1)*4 to jt_start_col + (condition_num-1)*4 + 3
                            base_col = jt_start_col + (condition_num - 1) * 4
                            
                            delay_perception = participant_row[base_col] if base_col < len(participant_row) else "N/A"
                            difficulty = participant_row[base_col + 1] if base_col + 1 < len(participant_row) else "N/A"
                            control_feeling = participant_row[base_col + 2] if base_col + 2 < len(participant_row) else "N/A"
                            embodiment = participant_row[base_col + 3] if base_col + 3 < len(participant_row) else "N/A"
                            
                            f.write(f"Latency {applied_latency} (Run {run_num}): → Condition {condition_num}\n")
                            f.write(f"  Did you experience delays between your actions and the robot's movements?: {delay_perception}\n")
                            f.write(f"  How difficult was it to perform the task?: {difficulty}\n") 
                            f.write(f"  I felt like I was controlling the movement of the robot: {control_feeling}\n")
                            f.write(f"  It felt like the robot was part of my body: {embodiment}\n\n")
                        else:
                            f.write(f"Latency {applied_latency} (Run {run_num}): → Could not map to condition\n\n")
                    
                else:
                    f.write(f"No questionnaire data found for participant {participant_num}\n")
                    
            except Exception as e:
                f.write(f"Error reading questionnaire file: {str(e)}\n")
        else:
            if not participant_num:
                f.write("Could not determine participant number from filename\n")
            if not os.path.exists(questionnaire_file):
                f.write(f"Questionnaire file not found: {questionnaire_file}\n")
    
    else:
        f.write("Could not find both ExpMarkers and LatencyMarkers streams for mapping.\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 80 + "\n")
    all_stream_names = [stream['info']['name'][0] if 'name' in stream['info'] else f"Stream {i+1}" for i, stream in enumerate(data)]
    marker_stream_names = [stream['info']['name'][0] if 'name' in stream['info'] else f"Stream {i+1}" for i, stream in enumerate(filtered_streams)]
    f.write(f"All available streams: {', '.join(all_stream_names)}\n")
    f.write(f"Analyzed marker streams: {', '.join(marker_stream_names)}\n")
    f.write("=" * 80 + "\n")

print("Analysis complete! Output written to 'stream_analysis_output.txt'")
print("(This script is for one participant mapping, not all)")

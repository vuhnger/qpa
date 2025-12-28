# User Studies: Robotic Hand Latency Perception

## Overview

This repository contains data and analysis code from a user study investigating human perception of latency in robotic hand control systems. The research examines how artificial latency affects user performance and subjective experience when controlling a robot arm through hand gestures.

## Experimental Setup

### Tasks
The experiment consisted of 2 standardized manual dexterity tasks:

1. **Jebsen-Taylor Hand Function Test (JTHFT) - Subtest 6**:
   - Moving 5 cans onto a board
   - Measurement: Time taken to complete the task

2. **Box & Block Test (BBT)**:
   - Moving blocks from one side of a box, over a barrier to the other side
   - Measurement: Number of blocks moved within 1 minute

### Experimental Conditions
- **5 latency conditions**: 0ms, 50ms, 100ms, 150ms, 200ms
- Each task was performed **5 times** (once per latency condition)
- Order of tasks and latency conditions was randomized for each participant
- One task was completed in all conditions before moving to the next task

### Data Collection
After each condition, participants answered questionnaires rating their experience on a 1-5 Likert scale:

1. **Delay Perception**: "Did you experience delays between your actions and the robot's movements?" (1 = minimal, 5 = significant)
2. **Task Difficulty**: "How difficult was it to perform the task?" (1 = very easy, 5 = impossible)
3. **Control Feeling**: "I felt like I was controlling the movement of the robot" (1 = strongly disagree, 5 = strongly agree)
4. **Embodiment**: "It felt like the robot was part of my body" (1 = strongly disagree, 5 = strongly agree)

Additional question: "How experienced are you with robotic systems?" (1 = beginner, 5 = expert)

## Dataset Structure

The dataset contains XDF (Extensible Data Format) files for each participant, recorded using [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder) with [LabStreamingLayer](https://github.com/sccn/labstreaminglayer).

### File Organization
```
datasets/
├── sub-001/
│   └── sub-001_ses-_task-_run-001.xdf
├── sub-002/
│   └── sub-002_ses-_task-_run-001.xdf
├── ...
├── sub-004/  # Special case: separate files for practice and main tasks
│   ├── sub-004_ses-_task-BBT_run-001.xdf
│   ├── sub-004_ses-_task-JTHFT_run-001_practice.xdf
│   └── sub-004_ses-_task-JTHFT_run-001.xdf
└── sub-026/
    └── sub-026_ses-_task-_run-001.xdf
```

### Data Streams
Each XDF file contains 4 data streams:

- **ExpMarkers**: Trial start/stop markers, block movement events in BBT, retry markers and reasons
- **LatencyMarkers**: Applied latency values for each condition
- **RokokoViveRaw**: Raw data from Rokoko motion tracking glove (hand and hip) + Vive tracker position/orientation
- **RokokoHandRaw**: Raw data from Rokoko glove with position/orientation of each finger joint

## Analysis Plan

We will analyze a subset of the questionnaire responses focusing on at least 4 questions from the study, with at least 1 question being from the participant questionnaires. The analysis will investigate:

- Threshold detection for latency perception at the hypothesized 100ms mark
- Performance degradation patterns across different latency conditions
- Correlation between objective performance measures and subjective ratings
- Individual differences in latency sensitivity

## Loading the Data

### Python
```python
import pyxdf

# Load XDF file
data, header = pyxdf.load_xdf('datasets/sub-001/sub-001_ses-_task-_run-001.xdf')

# Process streams
for stream in data:
    y = stream['time_series']
    name = stream['info']['name'][0]
    stream_type = stream['info']['type'][0]
    timestamps = stream['time_stamps']
```

## Data Quality Notes

Several participants experienced technical issues that may affect data quality:

| Participant | Issues |
|-------------|--------|
| sub-002 | JTHFT: 2nd latency condition skipped, condition 3 performed twice |
| sub-004 | Multiple XDF files (practice + main sessions) |
| sub-007 | Pink fingertip sensor lost, wrist tracking issues, lateral grip only for JTHFT |
| sub-008 | Missing pink fingertip sensor, static wrist position |
| sub-009 | Static wrist position, timing discrepancy in JTHFT trial 2 |
| sub-010 | Wrist tracking failed from JTHFT trial 3 onwards |

## Requirements

- Python packages: `pyxdf`, `numpy`, `pandas`, `matplotlib`, `scipy`
- MATLAB: EEGLAB with load_xdf plugin (alternative)

## Setup (macOS)

1. Install Python 3 (e.g., `brew install python`).
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running analysis scripts

- ANOVA (questionnaire latency): generates console output and `anova_results.csv`
  ```bash
  python run_anova_questions.py
  ```

- Questionnaire distributions plot (saves plot to file as configured in the script):
  ```bash
  python plot_questionnaire_distributions.py
  ```

- Post hoc latency pairwise tests (paired t-tests with Holm correction, writes `posthoc_latency_pairs.csv`):
  ```bash
  python posthoc_latency_tests.py
  ```
  - Also produces an optional compact visual of Holm-adjusted p-values for the first question: `posthoc_holm_pvalues_first_question.png` (bars by latency pair, red line at α=0.05).
  - And a Holm-adjusted p-value heatmap for the first question: `posthoc_holm_heatmap_first_question.png`.

- ANOVA visuals:
  - `anova_latency_means.png`: per-question latency means with SEM bars.
  - `anova_effect_sizes.png`: partial eta squared per question.

All visuals are saved to the `visuals/` folder (files are replaced if they already exist).

Ensure your terminal is inside the repository root before running the commands.

## Current results (questionnaire, latency)

Based on `run_anova_questions.py` and `posthoc_latency_tests.py`:

- Control feeling (`I felt like I was controlling the movement of the robot`): small but significant main effect of latency (F(4,120)=2.59, p=0.040, partial eta²≈0.08). Per-latency means drop at 200ms (≈3.77) relative to 0–150ms (≈4.13–4.23). Pairwise Holm-corrected tests did not reach significance; the largest raw differences are 0/50ms vs 200ms (raw p≈0.008–0.011, Holm p≈0.08–0.10).
- Embodiment (`It felt like the robot was part of my body`): no significant main effect (F(4,120)=0.58, p=0.68, partial eta²≈0.02). Means are broadly flat across latencies (≈2.74–3.00), and all pairwise contrasts are non-significant.

See `anova_results.csv` for omnibus stats and `posthoc_latency_pairs.csv` for pairwise details.

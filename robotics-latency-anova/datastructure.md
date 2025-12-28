# Latency Perception Data README

## The experiment
The experiment consisted of 2 tasks:
- Jebsen-Taylor Hand Function Test (JTHFT) - subtest 6: 
	- moving 5 cans onto a board
	- measurement: time taken to complete the task
- Box & Block Test (BBT): 
	- moving blocks from one side of the box, over a barrier to the other side
	- measurement: number of blocks moved within 1 minute

Both tasks were performed in 5 different conditions with artificial latency applied to the robotic system of 0, 50, 100, 150, and 200 ms. First one task was completed in all conditions, before moving on to the next task. The starting task as well as the order of conditions was randomised for each participant.

The questions that were asked after each condition were as follows:
1. Did you experience delays between your actions and the robot's movements?
	- 1 = minimal, 5 = significant
2. How difficult was it to perform the task?
	- 1 = very easy, 5 = impossible
3. I felt like I was controlling the movement of the robot
	- 1 = strongly disagree, 5 = strongly agree
4. It felt like the robot was part of my body
	- 1 = strongly disagree, 5 = strongly agree

For the general question "How experienced are you with robotic systems?", 1 is beginner, and 5 is expert.

## Data structure
From the experiment there will be two types of files. The first file will contain the questionnaire information of all participants. The other type will be an XDF file for each participant. The XDF format (Extensible Data Format) is a general-purpose format for time series and their meta-data that was jointly developed with LSL to ensure compatibility, see [here](http://github.com/sccn/xdf/). The data has been recorded by [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder), the default file recorder for use with [LabStreamingLayer](https://github.com/sccn/labstreaminglayer). LabRecorder records a collection of streams, including all their data (time-series / markers, meta-data) into a single XDF file.

The XDF files contain 4 streams consisting of the following data:
- *ExpMarkers*: 
	- contains the markers that indicate the start and stop of the trials, when a block is moved in the Box & Block Test, when a trial was retried and the reason for it.
- *LatencyMarkers*:
	- contains the data of which latency was applied for which condition.
- *RokokoViveRaw*: 
	- contains some raw data of the Rokoko motion tracking glove (hand and hip) and the position and orientation of the Vive tracker.
- *RokokoHandRaw*: 
	- contains the raw data of the Rokoko motion tracking glove, position and orientation of each finger joint.

Each stream contains time_series and time_stamps, where time_series is the information sent in a stream and time_stamps is the time_stamps of when this data was recorded.
## Loading the data

### Matlab
You can load the data in Matlab using [EEGLAB](https://github.com/xdf-modules/xdf-EEGLAB) and load_xdf plugin. Then, you can either use the EEGLAB gui or load the data in the command line using:
```
streams = load_xdf('file_name')
```

### Python
You can also load the data in Python using [pyxdf](https://github.com/xdf-modules/pyxdf/tree/5384490b96c96d0a2371541f4b50ee90e8c47b6f) :
```python
import pyxdf

data, header = pyxdf.load_xdf('file_name')

for stream in data:
	y = stream['time_series']
	name = stream['info']['name'][0]
	stream_type = stream['info']['type'][0]
	
	start = y[0]
	end = y[-1]
	number of samples = len(y)
```


## Other notes / observations / mistakes to take into account

While collecting data, the following notes / observations / mistakes were noted down that could influence the data analysis:

| Participant | Note                                                                                                                                                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 002         | In the 2nd task, JTHFT, the 2nd latency condition was accidentally skipped and latency condition 3 was done twice for trial 2 and 3.                                                                                                                                                       |
| 004         | Has 3 different XDF files: 1 for the practice session of JTHFT, 1 for the JTHFT trials, and 1 for the BBT.                                                                                                                                                                                 |
| 007         | The rubber of the pink finger tip fell off.<br>The wrist tracking was not working well, resulting in only using a lateral grip for JTHFT.<br>In BBT, the wrist was still unresponsive, but the palm is facing down now, so test is possible. In trial 3, the number of blocks should be 3. |
| 008         | Still missing rubber of the pink finger tip.<br>Wrist in static position but facing down.                                                                                                                                                                                                  |
| 009         | Wrist in static position but facing down.<br>In the 2nd trial of JTHFT, the end time should be the acceptance time instead of the end time.                                                                                                                                                |
| 010         | Wrist was working well until JTHFT trial 2. From 3 onwards, the wrist was straight.                                                                                                                                                                                                        |

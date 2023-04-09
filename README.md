# AimAide for CSGO and soon CS2

External realtime object detection-based aim aiding powered by <b>YOLOv8</b>, <b>CUDA</b> and <b>TensorRT</b>

<h3>Latest changes/additions</h3>
So far only Mirage is supported (although it might work on other maps depending on used agents).<br>

<h3>Hardware Requirements</h3>
To get this to work the detector has to run at 30fps at least.<br>
A NVIDIA GTX1070 non-TI with TensorRT runs at 30fps using the small model.

<h3>Installation</h3>
1) NVIDIA CUDA Toolkit >= 11.7<br>
2) Python 3.10.6 environment<br>
3) Corresponding PyTorch CUDA Package -> https://pytorch.org/get-started/locally/<br>
4) pip install -r requirements.txt<br><br>
Optional but recommended:<br>
5) NVIDIA TensorRT >= 8.4 -> https://developer.nvidia.com/tensorrt<br>

<h3>Usage</h3>
I) Disable windows mouse acceleration<br>
II) Disable raw input in CSGO<br>
III) Cap max_fps in CSGO at your native display refresh rate<br>
<br>
1) Run either run_tensorrt.py or run_yolo.py<br>
2) Selective detection can be activated by running with argument <b>-side 'your side'</b> (t, ct or dm for detecting all)<br>
If you want to change the detection mode while the script is running, simply write 't', 'ct' or 'dm' into the console and hit enter<br>
4) Depending on your hardware choose from 3 different models (nano, small, medium)<br>
nano (highest framerate, lowest detection performance),<br>
medium (lowest framerate, best decetion performance)<br>
5) Point at enemy and shoot

<h3>Benchmark mode</h3>
Run run_tensorrt.py or run_yolo.py with argument <b>-benchmark</b> to start in benchmark mode.<br>
This is going to run the detector in view-only- and detect-all mode for 300 iterations.<br>
Switch to CSGO and run/look around. At the end the average fps of the detector during that time will be displayed.


<h3>Arguments<h3>


| arg      | default   | Description                                                                                               |
| ----      | ---       | ---                                                                                                      |
| -input_size      | 640                                  | dimension of the input image for the detector                          |
| -engine or -weights | models/yolov8s_csgo_mirage-640-v5-al-gen-bg | selected engine (TensorRT) or weights (YOLOv8)                         |          
| -side            | 'dm'                                 | which side your are on, 'ct', 't' or 'dm' (deathmatch)                 | 
| -sensitivity     | 1                                    | sensitivity mode, increase when having a high framerate or chaotic aim |
| -visualize       | False                                | show live detector output in a new window                              |
| -view_only       | False                                | run in view only mode (disarmed)                                       |
| -benchmark       | False                                | launch benchmark mode                                                  |



<h3>FAQ</h3>

Q: Why is the aiming is so chaotic and unnatural?<br>
A: Probably due to high detector framerate. Increase the sensitivity mode by running with arg -sensitivity (default is 1)<br>
<br>
Q: Why is the aiming is so slow and laggy?<br>
A: Probably due to low detector framerate. Run benchmark mode and check if you get an average fps of at least 30 while being ingame.<br>
<br><br>
Feel free to fork this repo and/or use the models for your own projects. :)

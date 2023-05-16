# AimAide for CSGO and soon CS2

External realtime object detection-based aim aiding powered by <b>YOLOv8</b>, <b>CUDA</b> and <b>TensorRT</b><br>
[Twitter for further development updates](https://twitter.com/AimAideCS)

<img src="/docs/header_cts.jpg"><br>
[Video Demo here](https://github.com/schelmi1/AimAide/blob/main/docs/demo.mp4?raw=true)
<br>

<h3>Latest changes/additions</h3>


<br>
<b>16/05/23</b><br>
-new model which is optimized on heads on mirage (high headshot rate)<br>
models/yolov8s_csgo_mirage-320-v62-pal-gen-bg-head.pt<br>
-added flickieness argument to control how fast the mouse mover should flick to target
<br><br>
09/05/23 - bug in d3d_np grabber fixed (mixed up color channels), code improvements, removed engines from repo (engines will built locally),<br> d3d_gpu is disabled and needs to be rewritten<br>
16/04/23 - engine builder added to circumvent TensorRT incompatibilities <br>(by https://github.com/triple-Mu/YOLOv8-TensorRT)<br>
15/04/23 - introduced 320x320 input models which drastically increase fps with YOLO and TensorRT<br>


<h3>Supported Maps</h3>
* Mirage

<h3>Road Map</h3>
Models for CS2 and support for additional maps<br>
Human-like aim methods (like windmouse or ai-based)

<h3>Features</h3>
YOLOv8 Models trained on mirage with various CT and T agents (body and head).<br>
Simple smooth linear mouse mover locking onto target closest to crosshair.<br>

<h3>Hardware Requirements</h3>
To get this to work the detector has to run at 30fps at least.<br>
NVIDIA GTX1070 runs at 30fps on a 640x640 model or 60fps on a 320x320 model with TensorRT.<br>
NVIDIA RTX4090 should max out at ~120fps on a 640x640 model. (also TensorRT)<br>

<h3>Installation</h3>
1) NVIDIA CUDA Toolkit >= 11.7<br>
2) Python 3.10.6 environment<br>
3) Corresponding PyTorch CUDA Package -> https://pytorch.org/get-started/locally/<br>
4) pip install -r requirements.txt<br><br>
<b>Optional but recommended:</b><br>
5) NVIDIA TensorRT >= 8.4 -> https://developer.nvidia.com/tensorrt -> https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html<br> 
<br>
Speedup for bigger models with TensorRT is significant.<br>
<s>Thats why all models bigger than medium will only be released as TensorRT engines</s>.<br>
Models will be released as YOLO weights and locally built as TensorRT engines on first start up.<br>
This is due to TensorRT version incompatibilities.<br>

<img src="/docs/TensorRT_Speedup.png">

<h3>Usage</h3>
I) Disable windows mouse acceleration<br>
II) Disable raw input in CSGO<br>
III) Cap max_fps in CSGO at your native display refresh rate<br>
<br>
1) Run either run_tensorrt.py or run_yolo.py<br>
2) Selective detection can be activated by running with argument <b>-side 'your side'</b> (t, ct or dm for detecting all)<br>
If you want to change the detection mode while the script is running, simply write 't', 'ct' or 'dm' into the console and hit enter<br><br>
<img src="/docs/side_switch.png"><br>
3) Depending on your hardware choose from 3 different models (nano, small, medium)<br>
nano (highest framerate, lowest detection performance),<br>
medium (lowest framerate, best decetion performance)<br>
4) Run in benchmark mode first to see what framerate you get (over 60fps increase sensitivity mode)<br>
5) Adjust mouse sensitivity in CS and/or sensitivity mode of AimAide

<h3>Benchmark mode</h3>
Run run_tensorrt.py or run_yolo.py with argument <b>-benchmark</b> to start in benchmark mode.<br>
This is going to run the detector in view-only- and detect-all mode for 300 iterations.<br>
Switch to CSGO and run/look around. At the end the average fps of the detector during that time will be displayed.
<br><br>
<img src="/docs/benchmark_mode1.png">

<h3>Arguments<h3>


| arg      | default   | Description                                                                                               |
| ----      | ---       | ---                                                                                                      |
| --input_size      | 320                                  | dimension of the input image for the detector                          |
| --grabber      | 'win32'                                  | select screen grabber (win32, d3d_gpu, d3d_np)                          |
| --model           | models/yolov8s_csgo_mirage-320-v41-al-gen-bg | selected engine (TensorRT) or weights (YOLOv8)               |          
| --side            | 'dm'                                 | which side your are on, 'ct', 't' or 'dm' (deathmatch)                 | 
| --minconf         | 0.75                                  | minimum detection confidence                                           |  
| --sensitivity     | 1                                    | sensitivity mode, increase when having a high framerate or chaotic aim |
| --flickieness     | 4                                    | how flicky the mouse mover behaves (4 is slow, 16 is very flicky) |
| --visualize       | False                                | show live detector output in a new window                              |
| --view_only       | False                                | run in view only mode (disarmed)                                       |
| --benchmark       | False                                | launch benchmark mode                                                  |
| --no_engine_check | False                                | skips engine checking and building (run_tensorrt.py only)              |



<h3>FAQ</h3>
Q: Why does AimAide seem to be stuck on launch?<br>
A: This is a known issue with the YOLO class, run your command line as administrator.<br>
<br>
Q: Why is the aiming is so chaotic and unnatural?<br>
A: Probably due to high detector framerate. Increase the sensitivity mode by running with arg -sensitivity (default is 1)<br>
<br>
Q: Why is the aiming is so slow and laggy?<br>
A: Probably due to low detector framerate. Run benchmark mode and check if you get an average fps of at least 30 while being ingame.<br>
<br><br>
Feel free to fork this repo and/or use the models for your own projects. :)

# AimAide for CSGO and soon CS2
Realtime object detection-based aim aiding by using <b>YOLOv8</b>, <b>CUDA</b> and <b>TensorRT</b>

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
5) NVIDIA TensorRT -> https://developer.nvidia.com/tensorrt<br>

<h3>Usage</h3>
I) Disable windows mouse acceleration<br>
II) Disable raw input in CSGO<br>
III) Cap max_fps in CSGO at your native display refresh rate<br>
<br>
1) Run either run_tensorrt.py or run_yolo.py<br>
2) Selective detection can be activated by running with argument -side 'your side' (t, ct or dm for detecting all)<br>
If you want to change the detection mode while the script is running, simply write 't', 'ct' or 'dm' into the console and hit enter<br>
4) Depending on your hardware choose from 3 different models (nano, small, medium)<br>
nano (highest framerate, lowest detection performance),<br>
medium (lowest framerate, best decetion performance)<br>
5) Point at enemy and shoot

<h3>Arguments<h3>


<h3>FAQ</h3>

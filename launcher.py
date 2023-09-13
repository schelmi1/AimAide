import os
import ctypes
from threading import Event

import gradio as gr
from rich import print

from run_yolo import AimAideYolo

try:
    from run_tensorrt import AimAideTrt
    trt_err_flag: bool = False
except ImportError as e:
    trt_err_flag: bool = True

try:
    from src.enginebuilder import build_engine
except ImportError as e: ...


model_path = 'models/'
weights = [n for n in os.listdir(model_path) if n.lower().endswith('pt')]

def build_engines_initial(weights: str) -> None:
    for file in weights:
        engine_filename = file.replace('pt', 'engine')
        if engine_filename not in engines:
            inputsz = int(engine_filename.split('-')[1])
            print(f'[red]{engine_filename} is missing.[/red] [magenta]Building engine from YOLO weights. This may take a while...')
            pkl_filename = engine_filename.replace('engine', 'pkl')
            pt_filename =  engine_filename.replace('engine', 'pt')
            input_shape = (1, 3, inputsz, inputsz)
            build_engine(os.path.join(model_path, pt_filename), input_shape, os.path.join(model_path, pkl_filename))
            print(f'[yellow]Removing {pkl_filename}')
            os.remove(os.path.join(model_path, pkl_filename))
    print(f'[green]Done!')

if not trt_err_flag:
    try:
        engines = [n for n in os.listdir(model_path) if n.lower().endswith('engine')]
        if len(engines) != len(weights):
            print('[yellow]TensorRT found but engines are missing.\nBuilding engines from YOLO weights.')
            build_engines_initial(weights)
            engines = [n for n in os.listdir(model_path) if n.lower().endswith('engine')]
    except Exception as e:
        print('[red]Error checking/building engines!\nIf you encounter problems with TensorRT, use YOLOv8 inference.')
else:
    engines = ['TensorRT import error']
    
Aim = False

def run_aimaide(*args):
    global Aim, c
    infer, model, grabber, side, minconf, sense, flick, visual, viewonly, benchmark, custom_config, no_engine_check = args
    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
   
    inputsz = model.split('-')[1]
    if inputsz.isnumeric():
        inputsz = int(inputsz)

    side = 'dm' if side.lower() == 'deathmatch' else side.lower()

    sense = int(sense)
    flick = int(flick)

    if custom_config == '':
        model_config_path = os.path.join(model_path, 'config.json')
    else:
        model_config_path = os.path.join(custom_config)

    if infer.lower() == 'yolov8':
        Aim = AimAideYolo(screensz=(w, h), sectionsz=inputsz, grabber=grabber, infer_method='yolo', side=side,
                          model_path=os.path.join(model_path, model), 
                          model_config_path=model_config_path)
    elif infer.lower() == 'tensorrt':
        Aim = AimAideTrt(screensz=(w, h), sectionsz=inputsz, grabber=grabber, infer_method='trt', side=side,
                         model_path=os.path.join(model_path, model), 
                         model_config_path=model_config_path, no_engine_check=no_engine_check)
    
    if benchmark:
        Aim._benchmark()
    else:
        c = Event()
        Aim.run(minconf, sense, flick, visual, False, viewonly, False, c)

    Aim.listener_switch.join(timeout=1)
    if not Aim._grabber == 'win32':
        Aim.d.stop()

    return (infer, inputsz, model, side, minconf, sense, flick, visual, viewonly, custom_config)

def stop_aimaide():
    if not isinstance(Aim, bool):
        if Aim.running:
            Aim.exit(c)
    return 1

with gr.Blocks() as app:
    gr.Markdown(
        """
        # Welcome to AimAide for CS:GO!
        This Launcher is a QoL-feature for quickly adjusting settings.<br>
        Once you have selected an inference method, you can start AimAide via the run button.<br>
        Hit stop to halt AimAide.<br>
        <br>
        You can still run AimAide from CLI via run_*.py scripts!
    """
    )
    

    infer = gr.Radio(label="Inference", choices=["TensorRT", "YOLOv8"], info='TensorRT is recommended. If TensorRT is not working on your machine you can always fall back to YOLOv8.')
    model = gr.Dropdown(label="Model", choices=[], interactive=True, info='List of all available models in your /models folder')
    grabber = gr.Radio(label="Image grabber", choices=['win32', 'd3d_cpu', 'd3d_gpu'], value='win32', info='d3d_gpu is not working right now!')

    infer_map = {
        "YOLOv8": weights,
        "TensorRT": engines,
    }

    def filter_models(infer):
        if len(engines) == 0 and infer == 'TensorRT' and not trt_err_flag:
            build_engines_initial(weights)
            
        return gr.Dropdown.update(
            choices=infer_map[infer]
        ), gr.update(visible=True)

    def filter_value(model):
        if model in ():
            return gr.update(maximum=.99)
        else:
            return gr.update(maximum=.99)  

    with gr.Column(visible=True) as options:
        side = gr.Radio(label='Player side', choices=['CT', 'T', 'Deathmatch'], value='CT', info='Deathmatch locks onto CT and T.')
        minconf = gr.Slider(0.2, .99, value=.7, label='Minimum detection confidence', interactive=True, info='Lower value = more detections, more false positives | Higher value = less detections, less false positives')
        sense = gr.Slider(1, 16, label='Sensitivity Mode', interactive=True, info='Only increase if AimAide behaves erraticaly and fine-tuning in CS:GO is not fixing it.')
        flick = gr.Slider(4, 16, value=8, label='Flickieness', interactive=True, info='Lower Value = less flicky | Higher value = more flicky')
        with gr.Row():
            visual = gr.Checkbox(label='Show detector output', info='Opens a seperate window.')
            viewonly = gr.Checkbox(label='View only mode', info='Disables lock on target.')
            benchmark = gr.Checkbox(label='Benchmark mode', info='Runs in view only mode for 300 iterations and returns the average FPS.')
            no_engine_check = gr.Checkbox(label='No engine check', info='Disables engine checking when running with TensorRT. (check if you encounter problems with TensorRT while building engines)')
        custom_config = gr.Textbox(label="Path to custom model config file", info='If you want to use your own model, you can write a custom model config file and link it here. (see /models/config.json)') 

    with gr.Column(visible=False) as run_col:
        run_btn = gr.Button("Run")
        change_btn = gr.Button("Stop")
        output = gr.Textbox(label="Run settings")

    infer.change(filter_models, infer, [model, run_col])
    run_btn.click(run_aimaide, inputs=[infer, 
                                       model, 
                                       grabber, 
                                       side, 
                                       minconf,
                                       sense, 
                                       flick, 
                                       visual, 
                                       viewonly, 
                                       benchmark, 
                                       custom_config, 
                                       no_engine_check], outputs=[output])
    change_btn.click(stop_aimaide, outputs=[output])


    gr.Markdown("""
                
                Hints for best performance:<br>
                Run CSGO in your native refresh rate (fps_max)<br>
                Turn off Raw Input in CSGO<br>
                Turn off Mouse Acceleration in CSGO<br>
                Adjust Mouse Sensitivty in CSGO<br>
                """)

if __name__ == "__main__":
    app.launch()
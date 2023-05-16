import os
import sys
import ctypes
import argparse

import torch
import tensorrt
from rich import print

from src.enginebuilder import build_engine
from src.base import AimAide


class AimAideTrt(AimAide):
    def __init__(self, screensz: tuple, sectionsz: int, grabber: str, infer_method: str, 
                 model_path: str, model_config_path: str, side: str, no_engine_check: bool):
        if torch.cuda.is_available():
            print('[green]CUDA device found:', torch.cuda.get_device_name(0))
        else:
            print('[red]No CUDA device found.')
            sys.exit(0)

        print(f'[green]TensorRT found:[/green] [cyan]{tensorrt.__version__}')

        if model_path.endswith('pt'):
            print('[yellow]Specified YOLO.pt when a TensorRT engine is needed...Loading TensorRT engine!')
            model_path = model_path.replace('pt', 'engine')
        
        if not no_engine_check:
            print('[yellow]Checking engines...')
            rel_path = 'models/'
            weights = [n for n in os.listdir(rel_path) if n.endswith('pt')]
            engines = [n for n in os.listdir(rel_path) if n.endswith('engine')]
            for file in weights:
                engine_filename = file.replace('pt', 'engine')
                if engine_filename not in engines:
                    inputsz = int(engine_filename.split('-')[1])
                    print(f'[red]{engine_filename} is missing.[/red] [magenta]Building engine from YOLO weights. This may take a while...')
                    pkl_filename = engine_filename.replace('engine', 'pkl')
                    pt_filename =  engine_filename.replace('engine', 'pt')
                    input_shape = (1, 3, inputsz, inputsz)
                    build_engine(os.path.join(rel_path, pt_filename), input_shape, os.path.join(rel_path, pkl_filename))
                    print(f'[yellow]Removing {pkl_filename}')
                    os.remove(os.path.join(rel_path, pkl_filename))

        super().__init__(screensz, sectionsz, grabber, infer_method, model_path, model_config_path, side)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='AimAide with TensorRT', description='Run AimAide with TensorRT')
    parser.add_argument('--input_size', type=int, default=320,  
                        help='dimension of the input image for the detector')
    parser.add_argument('--grabber', type=str, default='win32', help='selected grabber (win32, d3d_gpu, d3d_np) ')
    parser.add_argument('--model', default='models/yolov8s_csgo_mirage-320-v41-al-gen-bg.engine', 
                        help='selected engine (TensorRT) ')
    parser.add_argument('--config', type=str, default='models/config.json', help='path to model config json')
    parser.add_argument('--side', type=str, default='dm', help='which side your are on, ct, t or dm (deathmatch))')
    parser.add_argument('--minconf', type=float, default=0.75, help='minimum detection confidence')
    parser.add_argument('--sensitivity' , type=int, default=1, 
                        help='sensitivity mode, increase when having a high framerate or chaotic aim')
    parser.add_argument('--flickieness' , type=int, default=4, 
                        help='how flicky the mouse mover behaves (4 is slow, 16 is very flicky)')
    parser.add_argument('--visualize', action='store_true', help='show live detector output in a new window')
    parser.add_argument('--view_only', action='store_true', help='run in view only mode (disarmed)')
    parser.add_argument('--benchmark', action='store_true', help='launch benchmark mode')
    parser.add_argument('--no_engine_check', action='store_true', help='avoid checking for missing engines')
    args = parser.parse_args()

    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    try:
        Aim = AimAideTrt((w, h), args.input_size, args.grabber, 'trt', args.model, args.config, args.side, args.no_engine_check)
    except AttributeError as e:
        print(e)
        print('[red]The selected engine is incompatible with your TensorRT version.\nDelete the engine from the models folder and run again to build a new engine from YOLO weights.')
        sys.exit()

    if args.benchmark:
        Aim._benchmark()
    else:
        Aim.run(args.minconf, args.sensitivity, args.flickieness, args.visualize, False, args.view_only, args.benchmark)

    Aim.listener_switch.join()

    if not Aim._grabber == 'win32':
        Aim.d.stop()
    ###
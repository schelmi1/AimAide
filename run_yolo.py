import sys
import ctypes
import argparse

import torch
from rich import print

from src.base import AimAide


class AimAideYolo(AimAide):
    def __init__(self, screensz: tuple, sectionsz: int, grabber: str, infer_method: str, 
                 model_path: str, model_config_path: str, side: str):
        if torch.cuda.is_available():
            print('[green]CUDA device found:', torch.cuda.get_device_name(0))
        else:
            print('[red]No CUDA device found.')
            sys.exit(0)

        if grabber == 'd3d_gpu':
            print('[yellow]d3d_gpu grabber is only available for AimAide with TensorRT, switching to d3d_np!')
            grabber = 'd3d_np'
        
        if model_path.endswith('engine'):
            print('[yellow]Specified TensorRT engine when YOLO.pt is needed...Loading YOLO weights!')
            model_path = model_path.replace('engine', 'pt')

        super().__init__(screensz, sectionsz, grabber, infer_method, model_path, model_config_path, side)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='AimAide with Ultralytics YOLO', description='Run AimAide with Ultralytics YOLO')
    parser.add_argument('--input_size', type=int, default=320,  
                        help='dimension of the input image for the detector')
    parser.add_argument('--grabber', type=str, default='win32', help='selected grabber (win32, d3d_gpu, d3d_np)')
    parser.add_argument('--model', default='models/yolov8s_csgo_mirage-320-v41-al-gen-bg.pt', 
                        help='selected weights (YOLO) ')
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
    args = parser.parse_args()

    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    Aim = AimAideYolo((w, h), args.input_size, args.grabber, 'yolo', args.model, args.config, args.side)

    if args.benchmark:
        Aim._benchmark()
    else:
        Aim.run(args.minconf, args.sensitivity, args.flickieness, args.visualize, False, args.view_only, args.benchmark)

    Aim.listener_switch.join()

    if not Aim._grabber == 'win32':
        Aim.d.stop()
    ####
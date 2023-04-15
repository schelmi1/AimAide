import sys
import ctypes
import argparse
from packaging import version

import torch
import tensorrt
from rich import print

from src.utils_trt import TRTModule
from src.base import AimAide


class AimAideTrt(AimAide):
    def __init__(self, screensz, sectionsz, grabber, model, side):
        super().__init__(screensz, sectionsz, grabber, side)
    
        if torch.cuda.is_available():
            print('[green]CUDA device found:', torch.cuda.get_device_name(0))
        else:
            print('[red]No CUDA device found.')
            sys.exit(0)

        print(f'TensorRT: {tensorrt.__version__}')
        
        if version.parse(tensorrt.__version__) < version.parse(str(8.4)):
            print('WARNING:\nEngines were built with TensorRT 8.4,\nits recommended to use at least TensorRT 8.4!')

        if model.endswith('pt'):
            print('[yellow]Specified YOLO.pt when a TensorRT engine is needed...Loading TensorRT engine!')
            model = model.replace('pt', 'engine')

        self.model = TRTModule(model, device=0)
        print(f'[green]Engine loaded:[/green] {model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='AimAide with TensorRT', description='Run AimAide with TensorRT')
    parser.add_argument('--input_size', type=int, default=320,  
                        help='dimension of the input image for the detector')
    parser.add_argument('--grabber', type=str, default='win32', help='selected grabber (win32, d3d_gpu, d3d_np) ')
    parser.add_argument('--model', default='models\yolov8s_csgo_mirage-320-v41-al-gen-bg.engine', 
                        help='selected engine (TensorRT) ')
    parser.add_argument('--side', type=str, default='dm', help='which side your are on, ct, t or dm (deathmatch))')
    parser.add_argument('--minconf', type=float, default=0.7, help='minimum detection confidence')
    parser.add_argument('--sensitivity' , type=int, default=1, 
                        help='sensitivity mode, increase when having a high framerate or chaotic aim')
    parser.add_argument('--visualize', action='store_true', help='show live detector output in a new window')
    parser.add_argument('--view_only', action='store_true', help='run in view only mode (disarmed)')
    parser.add_argument('--benchmark', action='store_true', help='launch benchmark mode')
    args = parser.parse_args()

    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    Aim = AimAideTrt((w, h), args.input_size, args.grabber, args.model, args.side)

    if args.benchmark:
        Aim._benchmark('trt')
    else:
        Aim.run('trt', args.minconf, args.sensitivity, args.visualize, False, args.view_only, args.benchmark)

    Aim.listener_switch.join()

    if not Aim._grabber == 'win32':
        Aim.d.stop()
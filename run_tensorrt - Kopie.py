import sys
import time
import ctypes
import argparse
from packaging import version
from threading import Thread

import cv2
import torch
import tensorrt
import numpy as np
from rich import print
from torchvision.ops import box_convert

from src.utils_trt import TRTModule, blob, letterbox, det_postprocess
from src.base import AimAide


class AimAideTrt(AimAide):
    def __init__(self, screensz, sectionsz, grabber, engine):
        super().__init__(screensz, sectionsz, grabber)
    
        if torch.cuda.is_available():
            print('CUDA device:', torch.cuda.get_device_name(0))
        else:
            print('No CUDA device found.')
            sys.exit(0)

        print(f'TensorRT: {tensorrt.__version__}')
        
        if version.parse(tensorrt.__version__) < version.parse(str(8.4)):
                print('WARNING:\nEngines were built with TensorRT 8.4,\nits recommended to use at least TensorRT 8.4!')

        print(f'Engine: {engine}')

        self.TrtEngine = TRTModule(engine, device=0)
        self.__TrTEngine_W, self.__TrTEngine_H = self.TrtEngine.inp_info[0].shape[-2:]

    def run(self, prefer_body, side, minconf, sensitivity, visualize, view_only, 
            benchmark_mode=False, benchmark_iters=-1):
        
        self.conf = 0
        count = 0
        count_fps = 0
        dx, dy = 0, 0
        maxdist = np.hypot(self.section_size, self.section_size)

        self._switch_side(side)
        self.listener_switch = Thread(target=self.user_switch_side, daemon=True)
        self.listener_switch.start()
        
        while True:
            runtime_start = time.perf_counter()
            
            if self._grabber == 'd3d_gpu':
                tensor = self._grab_d3d_gpu()
                ratio = 1.
                dwdh = torch.tensor([0., 0., 0., 0.], device=0)
            
            if self._grabber == 'win32':
                img = self._grab()

            if self._grabber == 'd3d_np':
                img = self._grab_d3d_np()

            if self._grabber == 'd3d_np' or self._grabber == 'win32':
                bgr, ratio, dwdh = letterbox(img, (self.__TrTEngine_W, self.__TrTEngine_H))
                tensor = blob(bgr[:, :, ::-1], return_seg=False)
                dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=0)
                tensor = torch.asarray(tensor, device=0)

            data = self.TrtEngine(tensor)
            bboxes, confs, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio
            
            if len(bboxes) > 0:
                bboxes = box_convert(bboxes,'xyxy', 'cxcywh')
            bboxes =  bboxes.squeeze().cpu().numpy().astype(np.intc)
            confs = confs.squeeze().cpu().numpy().tolist()
            labels = labels.squeeze().cpu().numpy().tolist()

            if isinstance(confs, float):
                confs = [confs]
            if isinstance(labels, int):
                labels = [labels]

            confs = np.array(confs, dtype=np.float64)
            labels = np.array(labels, dtype=np.uintc)

            if bboxes.ndim == 1:
                bboxes = bboxes[None, :]

            x1, y1, x2, y2, w, h, dx, dy = self.perform_target_selection(bboxes, confs, labels, prefer_body=prefer_body)

            if self.detected and self.conf > minconf and np.hypot(dx, dy) < maxdist and not view_only:
                self._smooth_linear_aim(dx, dy, [x1, y1, x2, y2, w, h], sensitivity)

            if visualize:
                if self._grabber == 'd3d_gpu':
                    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                plot = self.visualize(img.astype(np.uint8), bboxes, confs, labels, minconf)
                cv2.imshow('cap', plot)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            runtime_end = time.perf_counter()        
            if count_fps == 30:
                print(f'[{self._side_color}]FPS:{(int(1/(runtime_end-runtime_start)))} | conf:{self.conf:.2f} dx:{abs(dx):03d} dy:{abs(dy):03d}' , end='\r', flush=True)
                count_fps = 0
            else:
                count_fps += 1

            if benchmark_mode:
                count += 1
                self._avg_fps.append(int(1/(runtime_end-runtime_start)))
                if count > benchmark_iters:
                    print(f'Average FPS: {np.mean(np.array(self._avg_fps)):.2f} (dev:{np.std(np.array(self._avg_fps)):.2f})                       _')
                    print('Above 120 FPS increase sensitivity if aim aiding feels unnatural (-sensitivity argument: default 1).')
                    break
        sys.exit()

    def benchmark(self, iters=300):
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 5:
            t = time.perf_counter() - t1
            print(f'Running in benchmark mode for {iters} iterations in {(4-t):.1f} sec!', end='\r', flush=True)
            time.sleep(1)
        print('\n')
        self._avg_fps = []
        self.run(False, side='dm', minconf=.8, sensitivity=1, visualize=False,
                 view_only=True ,benchmark_mode=True, benchmark_iters=iters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='AimAide with TensorRT', description='Run AimAide with TensorRT')
    parser.add_argument('--input_size', type=int, default=640,  
                        help='dimension of the input image for the detector')
    parser.add_argument('--grabber', type=str, default='win32', help='selected grabber (win32, d3d_gpu, d3d_np) ')
    parser.add_argument('--engine', default='models\yolov8s_csgo_mirage-640-v5-al-gen-bg.engine', 
                        help='selected engine (TensorRT) ')
    parser.add_argument('--side', type=str, default='dm', help='which side your are on, ct, t or dm (deathmatch))')
    parser.add_argument('--minconf', type=float, default=0.8, help='minimum detection confidence')
    parser.add_argument('--sensitivity' , type=int, default=1, 
                        help='sensitivity mode, increase when having a high framerate or chaotic aim')
    parser.add_argument('--visualize', action='store_true', help='show live detector output in a new window')
    parser.add_argument('--view_only', action='store_true', help='run in view only mode (disarmed)')
    parser.add_argument('--benchmark', action='store_true', help='launch benchmark mode')
    args = parser.parse_args()

    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    Aim = AimAideTrt((w, h), args.input_size, args.grabber, args.engine)

    if args.benchmark:
        Aim.benchmark()
    else:
        Aim.run(False, args.side, args.minconf, args.sensitivity, args.visualize, args.view_only)

    Aim.listener_switch.join()

    if not Aim._grabber == 'win32':
        Aim.d.stop()
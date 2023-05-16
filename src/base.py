
import os
import sys
import time
from typing import Union
from threading import Thread

try:
    import d3dshot
    d3d_err_flag: bool = False
except ImportError as e:
    print(e, '...running win32 grabber')
    d3d_err_flag: bool = True

import cv2
import torch 
import pyautogui
import numpy as np
from rich import print
from ultralytics import YOLO
import win32gui, win32ui, win32con
from torchvision.ops import box_convert

from .utils import accurate_timing, parse_config

try:
    from .utils_trt import blob, letterbox, det_postprocess, TRTModule
except ImportError as e:
    print(f'[yellow]{e}, only YOLO inference available!')

MOUSE_X_MOVE_DAMPING: float = 1
MOUSE_Y_MOVE_DAMPING: float = 1.3

SIDE_COLOR_MAP = {'ct' : 'turquoise2',
                  't'  : 'yellow',
                  'dm' : 'green'
                 }

class AimAide():
    def __init__(self, screensz: tuple[int, int], sectionsz: int, grabber: str, infer_method: str,  
                 model_path: str, model_config_path: str, side: str) -> None:
        model_size = os.path.basename(model_path).split('-')[1]
        inputsz = int(model_size) if model_size.isnumeric() else model_size
        if isinstance(inputsz, int) and inputsz != sectionsz:
            print(f'[red]Model input size and grabber size are not the same.\nModel: {inputsz}, Grabber: {sectionsz}\nChange the grabber size by using the --input_size argument.')
        elif not isinstance(inputsz, int):
            print('[red]Unknown model name format.')

        self.side = side
        self._side_color = SIDE_COLOR_MAP[side]

        self.screen_width, self.screen_height = screensz

        assert isinstance(self.screen_width, int)
        assert isinstance(self.screen_height, int)

        self.section_size = sectionsz
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        try:
            if grabber == 'd3d_gpu' and not d3d_err_flag:
                print('[red]D3D GPU GRABBING IS CURRENTLY NOT WORKING AND WILL BE REWRITTEN.\nUse win32 or d3d_np!')
                self._grabber = 'd3d_gpu'
                self.d = d3dshot.create(capture_output="pytorch_gpu")
                self.d.display = self.d.displays[0]
                sys.exit()

            if grabber == 'd3d_np' and not d3d_err_flag:
                self._grabber = 'd3d_np'
                self.d = d3dshot.create(capture_output="numpy")
                self.d.display = self.d.displays[0]
                self._grabfunc = self._grab_d3d_np

            if grabber == 'win32' or d3d_err_flag:
                self._grabber = 'win32'
                self.dcObj = win32ui.CreateDCFromHandle(win32gui.GetWindowDC(win32gui.GetDesktopWindow()))
                self.cDC = self.dcObj.CreateCompatibleDC()
                self.dataBitMap = win32ui.CreateBitmap()
                self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.section_size, self.section_size)
                self.cDC.SelectObject(self.dataBitMap)
                self._grabfunc = self._grab

            print(f'[green]Grabber started:[/green] {self._grabber}')
            
            if infer_method == 'trt':
                self.model = TRTModule(model_path, device=0)
                self._infer_func = self._inference_trt

            if infer_method == 'yolo':
                self.model = YOLO(model_path)
                self._infer_func = self._inference_yolo

            print(f'[green]Model loaded:[/green] {model_path}')

        except Exception as e:
            print(e)
            sys.exit()
    
        ct, t, body_ct, body_t = parse_config(model_config_path)
        self.side_target_label_map = {'ct':t, 't':ct, 'body_ct':body_t, 'body_t':body_ct, 'dm':ct + t}
        print(f'[green]Model config file loaded[/green]: {model_config_path}')

    def _switch_side(self, side: str) -> None:
        if side in self.side_target_label_map:
            self.side = side
        else:
            self.side = 'dm'

        print(f'[magenta]Switching side to: {self.side}')
        self._side_color = SIDE_COLOR_MAP[self.side]
        print(end='\n')

    def user_switch_side(self) -> None:
        while True:
            user_in = str(input(''))
            if user_in in self.side_target_label_map:
                self._switch_side(user_in)
            else:
                print(f'[red]Bad input [bold]{user_in}[/bold]. Type ct, t or dm!')

    def _grab(self) -> np.ndarray:
        try: 
            self.cDC.BitBlt((0, 0), (self.section_size, self.section_size), self.dcObj, 
                            (self.center_x-self.section_size//2, self.center_y-self.section_size//2), 
                            win32con.SRCCOPY)

            img = np.frombuffer(self.dataBitMap.GetBitmapBits(True), dtype='uint8')
            img.shape = (self.section_size, self.section_size, 4)
            img = img[..., :3]

            return img#cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(e)
            sys.exit(0)
    
    def _grab_d3d_gpu(self) -> torch.Tensor:
        try:
            tensor = self.d.screenshot(region=(self.screen_width//2 - self.section_size//2, 
                                               self.screen_height//2 - self.section_size//2, 
                                               self.screen_width//2 + self.section_size//2, 
                                               self.screen_height//2 + self.section_size//2)
                                       )
            tensor = torch.flip(tensor, dims=(2, )).permute(2, 0, 1)[None, :, :, :]
            
            return tensor.type(torch.float32) / 255

        except Exception as e:
            print(e)
            sys.exit(0)

    def _grab_d3d_np(self) -> np.ndarray:
            img = self.d.screenshot(region=(self.screen_width//2 - self.section_size//2, 
                                            self.screen_height//2 - self.section_size//2, 
                                            self.screen_width//2 + self.section_size//2, 
                                            self.screen_height//2 + self.section_size//2)
                                    )
            
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    def _inference_yolo(self, img: np.ndarray)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self.model.predict(img, device=0, verbose=False, max_det=10)

        bboxes, confs, labels = [], [], []
        for d in reversed(results[0].boxes):
            bboxes.append(d.xywh.squeeze().cpu().numpy())
            confs.append(float(d.conf.squeeze().cpu()))
            labels.append(int(d.cls.squeeze().cpu()))
    
        bboxes = np.array(bboxes, dtype=np.uintc)
        confs = np.array(confs, dtype=np.float64)
        labels = np.array(labels, np.uintc)

        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]

        return (bboxes, confs, labels)

    def _inference_trt(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._grabber == 'd3d_np' or self._grabber == 'win32':
            bgr, ratio, dwdh = letterbox(img, (self.section_size, self.section_size))
            tensor = blob(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=0)
            tensor = torch.asarray(tensor, device=0)

        data = self.model(tensor)
        bboxes, confs, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio
        
        if len(bboxes) > 0:
            bboxes = box_convert(bboxes,'xyxy', 'cxcywh')
        bboxes =  bboxes.squeeze().cpu().numpy().astype(np.uintc)
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

        return (bboxes, confs, labels)

    def _perform_target_selection(self, 
                                  bboxes: np.ndarray, 
                                  confs: np.ndarray, 
                                  labels: np.ndarray, 
                                  prefer_body: bool) -> tuple[bool, float, int, int, int, int, int, int, int, int]:
        target = []
        if labels.size > 1:
            valid_idcs = np.in1d(labels, self.side_target_label_map[self.side])
            bboxes = bboxes[valid_idcs]
            confs = confs[valid_idcs]
            labels = labels[valid_idcs]
            dist = bboxes[:, 0] - self.section_size // 2
            right_sided = np.where(dist > 0)[0]
            if len(right_sided) > 0:
                dist = dist[right_sided]
                bboxes = bboxes[right_sided]
                confs = confs[right_sided]
                labels = labels[right_sided]             
            closest = np.argsort(dist)
            
            if (self.side_target_label_map['body_ct'] or self.side_target_label_map['body_t']) in labels and (prefer_body):
                closest_body_idx = np.where((labels[closest] == self.side_target_label_map['body_ct']).any() or (labels[closest] == self.side_target_label_map['body_t']).any())[0][0]
                if len(closest_body_idx) > 0:
                    target_body_idx = closest[closest_body_idx]
                    target = bboxes[target_body_idx]
                    conf = float(confs[target_body_idx])

            elif len(closest) > 0:
                target = bboxes[closest[0]]
                conf = float(confs[closest[0]])
        
        elif labels.size == 1 and labels in self.side_target_label_map[self.side]:
            target = bboxes
            conf = float(confs)

        if len(target) > 0:
            cx, cy, w, h = np.squeeze(target)
            x1, x2 = (cx - w//2 - self.section_size // 2, cx + w//2 - self.section_size // 2)
            y1, y2 = (cy - h//2 - self.section_size // 2, cx + h//2  - self.section_size // 2)
            dx = int(cx-self.section_size//2)
            dy = int(cy-self.section_size//2)

            return (True, conf, x1, y1, x2, y2, w, h, dx, dy)
        else:
            return (False, float(0), 0, 0, 0, 0, 0, 0, 0, 0)

    def _smooth_linear_aim(self, dx: int, dy: int, xyxywh: list, sensitivity: int, flickieness: int) -> None:
            x1, y1, x2, y2, w, h = xyxywh
            n_steps = abs(dx//flickieness) if abs(dx) > 100 else abs(dx//flickieness)
            if n_steps < 1:
                n_steps = 1
            for _ in range(n_steps):
                smooth = 2 * n_steps * sensitivity
                if (x1+w//3 > 0) or (x2-w//3 < 0) or (y1-h//4 > 0) or (y2+h//4 < 0):
                    pyautogui.move(int(dx//smooth/MOUSE_X_MOVE_DAMPING), int(dy//smooth/MOUSE_Y_MOVE_DAMPING), 0, _pause=False)
                _ = accurate_timing(sensitivity)
    
    def _visualize(self, 
                   frame: Union[np.ndarray, torch.Tensor], 
                   bboxes: Union[np.ndarray, list], 
                   confs: Union[np.ndarray, list], 
                   labels: Union[np.ndarray, list], 
                   min_conf: float) -> np.ndarray:

        if isinstance(frame, torch.Tensor):
            frame = frame.squeeze().permute(1, 2, 0).cpu().numpy() * 255
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        for bbox, conf, label in zip(bboxes, confs, labels):
            cx, cy, w, h = bbox
            conf = round(conf, 2)
            if conf > min_conf:
                if label == 0 or label == 1:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 255, 0), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 255, 9), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)
                elif label == 2 or label == 3:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 0, 255), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 0, 255), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)

        return frame

    def _benchmark(self) -> None:
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 5:
            t = time.perf_counter() - t1
            print(f'Running in benchmark mode for 300 iterations in {(4-t):.1f} sec!', end='\r', flush=True)
            time.sleep(1)
        print('\n')
        self.run(min_conf=.8, visualize=False, prefer_body=False, sensitivity=1, view_only=True, benchmark=True)


    def run(self, min_conf: float, sensitivity: int, flickness: int, 
            visualize: bool, prefer_body: bool, view_only: bool, benchmark: bool) -> None:

        self.listener_switch = Thread(target=self.user_switch_side, daemon=True)
        self.listener_switch.start()
        
        max_dist = np.hypot(self.section_size, self.section_size)
        conf = 0
        count_fps, count = 0, 0
        dx, dy = 0, 0
        
        avg_fps = []
        while True:
            runtime_start = time.perf_counter()
            img = self._grabfunc()
            bboxes, confs, labels = self._infer_func(img)
            detected, conf, x1, y1, x2, y2, w, h, dx, dy = self._perform_target_selection(bboxes, confs, labels, prefer_body=prefer_body)

            if detected and conf > min_conf and np.hypot(dx, dy) < max_dist and not view_only:
                self._smooth_linear_aim(dx, dy, [x1, y1, x2, y2, w, h], sensitivity, flickness)

            runtime_end = time.perf_counter()        
            if count_fps == 30:
                print(f'[{self._side_color}]FPS:{(int(1/(runtime_end-runtime_start)))} | conf:{conf:.2f} dx:{abs(dx):03d} dy:{abs(dy):03d}' , end='\r', flush=True)
                count_fps = 0
            else:
                count_fps += 1

            if visualize:
                plot = self._visualize(img, bboxes, confs, labels, min_conf)
                cv2.imshow('cap', plot)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if benchmark:
                count += 1
                avg_fps.append(int(1/(runtime_end-runtime_start)))
                if count > 300:
                    print(f'Average FPS: {np.mean(np.array(avg_fps)):.2f} (dev:{np.std(np.array(avg_fps)):.2f})                       _')
                    print('Above 60 FPS increase sensitivity if aim aiding feels unnatural (-sensitivity argument: default 1).')
                    break
        sys.exit()
        ###
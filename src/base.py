global d3d_err_flag

import sys
import time
from typing import Union
from threading import Thread

try:
    import d3dshot
    d3d_err_flag = False
except ImportError as e:
    print(e, '...running win32 grabber')
    d3d_err_flag = True

import cv2
import torch 
import pyautogui
import numpy as np
from rich import print
import win32gui, win32ui, win32con
from torchvision.ops import box_convert

from .utils import accurate_timing

try:
    from .utils_trt import blob, letterbox, det_postprocess
except ImportError as e:
    print(e, 'only YOLO inference available!')


MOUSE_X_MOVE_DAMPING = 1
MOUSE_Y_MOVE_DAMPING = 1.3

SIDE_TARGET_LABEL_MAP = {'ct' : np.array([2, 3]),
                          't' : np.array([0, 1]),
                         'dm' : np.array([0, 1, 2, 3])
                        }

SIDE_COLOR_MAP = {'ct' : 'turquoise2',
                  't'  : 'yellow',
                  'dm' : 'green'
                 }

BODY_CT = 0
BODY_T = 2

class AimAide():
    def __init__(self, screensz: int, sectionsz: int, grabber: str, side: str) -> None:        
        self.side = side
        self._switch_side(self.side)

        self.screen_width = screensz[0]
        self.screen_height = screensz[1]
        self.section_size = sectionsz
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        try:
            if grabber == 'd3d_gpu' and not d3d_err_flag:
                self._grabber = 'd3d_gpu'
                self.d = d3dshot.create(capture_output="pytorch_gpu")
                self.d.display = self.d.displays[0]

            if grabber == 'd3d_np' and not d3d_err_flag:
                self._grabber = 'd3d_np'
                self.d = d3dshot.create(capture_output="numpy")
                self.d.display = self.d.displays[0]

            if grabber == 'win32' or d3d_err_flag:
                self._grabber = 'win32'
                self.dcObj = win32ui.CreateDCFromHandle(win32gui.GetWindowDC(win32gui.GetDesktopWindow()))
                self.cDC = self.dcObj.CreateCompatibleDC()
                self.dataBitMap = win32ui.CreateBitmap()
                self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.section_size, self.section_size)
                self.cDC.SelectObject(self.dataBitMap)
        except Exception as e:
            print(e)
            sys.exit()
                   
        print(f'[green]Grabber started:[/green] {self._grabber}')

    def _switch_side(self, side: str) -> None:
        self.side = side
        self._side_color = SIDE_COLOR_MAP[side]

    def user_switch_side(self) -> None:
        while True:
            user_in = str(input(''))
            if user_in in SIDE_TARGET_LABEL_MAP:
                print(f'[magenta]Switching side to: {user_in}')
                self._switch_side(user_in)
                print(end='\n')
            else:
                print(f'[red]Bad input {user_in}. Type ct, t or dm!')

    def _grab(self) -> np.ndarray:
        try: 
            self.cDC.BitBlt((0, 0), (self.section_size, self.section_size), self.dcObj, 
                            (self.center_x-self.section_size//2, self.center_y-self.section_size//2), 
                            win32con.SRCCOPY)

            img = np.frombuffer(self.dataBitMap.GetBitmapBits(True), dtype='uint8')
            img.shape = (self.section_size, self.section_size, 4)
            img = img[..., :3]

            return img
            
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
            
            return img

    def _inference_yolo(self)-> tuple([np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        if self._grabber == 'win32':
            img = self._grab()

        if self._grabber == 'd3d_np':
            img = self._grab_d3d_np()

        results = self.model.predict(img, device=0, verbose=False, max_det=10)

        bboxes, confs, labels = [], [], []
        for d in reversed(results[0].boxes):
            bboxes.append(d.xywh.squeeze().cpu().numpy().astype(int))
            confs.append(float(d.conf.squeeze().cpu()))
            labels.append(int(d.cls.squeeze().cpu()))
    
        bboxes = np.array(bboxes, dtype=np.uintc)
        confs = np.array(confs, dtype=np.float64)
        labels = np.array(labels, np.uintc)

        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]

        return (img, bboxes, confs, labels)

    def _inference_trt(self) -> tuple([Union[np.ndarray, torch.Tensor], np.ndarray, np.ndarray, np.ndarray]):
        if self._grabber == 'd3d_gpu':
            tensor = self._grab_d3d_gpu()
            ratio = 1.
            dwdh = torch.tensor([0., 0., 0., 0.], device=0)
            img = tensor
        
        if self._grabber == 'win32':
            img = self._grab()

        if self._grabber == 'd3d_np':
            img = self._grab_d3d_np()

        if self._grabber == 'd3d_np' or self._grabber == 'win32':
            bgr, ratio, dwdh = letterbox(img, (self.section_size, self.section_size))
            tensor = blob(bgr[:, :, ::-1], return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=0)
            tensor = torch.asarray(tensor, device=0)

        data = self.model(tensor)
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

        return (img, bboxes, confs, labels)

    def _perform_target_selection(self, 
                                  bboxes: np.ndarray, 
                                  confs: np.ndarray, 
                                  labels: np.ndarray, 
                                  prefer_body: bool) -> tuple[int, int, int, int, int, int, int, int]:
        
        conf = float(0)
        target = []

        if labels.size > 1:
            valid_idcs = np.in1d(labels, SIDE_TARGET_LABEL_MAP[self.side])
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
            
            if (BODY_CT or BODY_T) in labels and (prefer_body):
                closest_body_idx = np.where((labels[closest] == BODY_CT).any() or (labels[closest] == BODY_T).any())[0][0]
                target_body_idx = closest[closest_body_idx]
                target= bboxes[target_body_idx]
                conf = float(confs[target_body_idx])
            elif len(closest) > 0:
                target = bboxes[closest[0]]
                conf = float(confs[closest[0]])
        
        elif labels.size == 1 and labels in SIDE_TARGET_LABEL_MAP[self.side]:
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
            return (False, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _smooth_linear_aim(self, dx: int, dy: int, xyxywh: list, sensitivity: int) -> None:
            x1, y1, x2, y2, w, h = xyxywh
            n_steps = abs(dx//4) if abs(dx) > 100 else abs(dx//6)
            if n_steps <= 3:
                n_steps = 2
            for _ in range(n_steps):
                smooth = 2 * n_steps * sensitivity
                if (x1+w//4 > 0) or (x2-w//4 < 0) or (y1-h//8 > 0) or (y2+h//8 < 0):
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

    def _benchmark(self, infer_method: str) -> None:
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 5:
            t = time.perf_counter() - t1
            print(f'Running in benchmark mode for 300 iterations in {(4-t):.1f} sec!', end='\r', flush=True)
            time.sleep(1)
        print('\n')
        self.run(infer_method, min_conf=.8, visualize=False, prefer_body=False, sensitivity=1, view_only=True, benchmark=True)


    def run(self, infer_method: str, min_conf: float, sensitivity: int, 
            visualize: bool, prefer_body: bool, view_only: bool, benchmark: bool) -> None:

        self.listener_switch = Thread(target=self.user_switch_side, daemon=True)
        self.listener_switch.start()
        
        max_dist = np.hypot(self.section_size, self.section_size)
        conf = 0
        count_fps, count = 0, 0
        dx, dy = 0, 0

        if infer_method == 'trt':
            _infer_func = self._inference_trt
        if infer_method == 'yolo':
            _infer_func = self._inference_yolo

        avg_fps = []
        while True:
            runtime_start = time.perf_counter()
            img, bboxes, confs, labels = _infer_func()
            detected, conf, x1, y1, x2, y2, w, h, dx, dy = self._perform_target_selection(bboxes, confs, labels, prefer_body=prefer_body)

            if detected and conf > min_conf and np.hypot(dx, dy) < max_dist and not view_only:
                self._smooth_linear_aim(dx, dy, [x1, y1, x2, y2, w, h], sensitivity)

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
global d3d_err_flag

import sys
import cv2
import numpy as np
try:
    import d3dshot
    d3d_err_flag = False
except ImportError as e:
    print(e, '...running win32 grabber')
    d3d_err_flag = True
import torch
import win32gui, win32ui, win32con
from typing import Union

class AimAide():
    def __init__(self, screensz: int, sectionsz: int, grabber: str) -> None:        
        self.side = 'dm'
        if self.side == 'dm':
            self._side_color = 'green'
        if self.side == 'ct':
            self._side_color = 'turquoise2'
        if self.side == 't':
           self._side_color = 'yellow'

        self.screen_width = screensz[0]
        self.screen_height = screensz[1]
        self.section_size = sectionsz
        self.center_x = self.screen_width // 2
        self.center_y = self.screen_height // 2
        
        if grabber == 'd3d_gpu' and not d3d_err_flag:
            self._grabber = grabber
            print(f'Starting grabber: {self._grabber}')
            self.d = d3dshot.create(capture_output="pytorch_gpu")
            self.d.display = self.d.displays[0]

        if grabber == 'd3d_np' and not d3d_err_flag:
            self._grabber = grabber
            print(f'Starting grabber: {self._grabber}')
            self.d = d3dshot.create(capture_output="numpy")
            self.d.display = self.d.displays[0]

        else:
            self._grabber = 'win32'
            print(f'Starting grabber: {self._grabber}')
            self.dcObj = win32ui.CreateDCFromHandle(win32gui.GetWindowDC(win32gui.GetDesktopWindow()))
            self.cDC = self.dcObj.CreateCompatibleDC()
            self.dataBitMap = win32ui.CreateBitmap()
            self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.section_size, self.section_size)

    def _switch_side(self, side: str) -> None:
        self.side = side
        if side == 'ct':
            self._side_color = 'turquoise2'
        if side == 't':
            self._side_color = 'yellow'
        if side == 'dm':
            self._side_color = 'green'

    def user_switch_side(self) -> None:
        while True:
            self.side = str(input(''))
            self._switch_side(self.side)
            print(end='\n')

    def perform_target_acquisition(self):
        pass

    def processing_trt_tensor(self) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    def processing_trt(self) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    def processing_yolo(self) -> tuple[np.ndarray, np.ndarray, float]:
        pass

    def _grab(self) -> np.ndarray:
        try: 
            self.cDC.SelectObject(self.dataBitMap)
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
    
    def _grab_d3d_gpu(self) -> torch.tensor:
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

    def visualize(self, frame: np.ndarray, 
                        bboxes: Union[np.ndarray, list], 
                        confs: Union[np.ndarray, list], 
                        labels: Union[np.ndarray, list], 
                        minconf: float) -> np.ndarray:
        frame =  np.ascontiguousarray(frame, dtype=np.uint8)
        for bbox, conf, label in zip(bboxes, confs, labels):
            cx, cy, w, h = bbox
            conf = round(conf, 2)
            if conf > minconf:
                if label == 0 or label == 1:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 255, 0), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 255, 9), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)
                elif label == 2 or label == 3:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 0, 255), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 0, 255), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)

        return frame

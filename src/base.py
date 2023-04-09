import cv2
import numpy as np
import win32gui, win32ui

class AimAide():
    def __init__(self, screensz, sectionsz):        
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

        self.dcObj = win32ui.CreateDCFromHandle(win32gui.GetWindowDC(win32gui.GetDesktopWindow()))
        self.cDC = self.dcObj.CreateCompatibleDC()
        self.dataBitMap = win32ui.CreateBitmap()
        self.dataBitMap.CreateCompatibleBitmap(self.dcObj, self.section_size, self.section_size)

    def _switch_side(self, side):
        self.side = side
        if side == 'ct':
            self._side_color = 'turquoise2'
        if side == 't':
            self._side_color = 'yellow'
        if side == 'dm':
            self._side_color = 'green'

    def user_switch_side(self):
        while True:
            self.side = str(input(''))
            self._switch_side(self.side)
            print(end='\n')

    def perform_target_acquisition(self):
        pass

    def visualize(self, frame, bboxes, confs, labels):
        frame =  np.ascontiguousarray(frame, dtype=np.uint8)
        for bbox, conf, label in zip(bboxes, confs, labels):
            cx, cy, w, h = bbox
            conf = round(conf, 2)
            if conf > .8:
                if label == 0 or label == 1:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 255, 0), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 255, 9), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)
                elif label == 2 or label == 3:
                    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), color=(255, 0, 255), thickness=2)
                    cv2.putText(frame, str(conf), (cx, cy), color=(255, 0, 255), fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, thickness=2)

        return frame

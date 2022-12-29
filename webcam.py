import cv2 
import numpy as np
import time

class Webcams:
  
    def __init__(self):
        self.DEVICES = []
        for i in range(5):
           cap = cv2.VideoCapture(i)
           if cap is None or not cap.isOpened():
                continue
           self.DEVICES.append(i)

    def device_list(self):
        return self.DEVICES  

if __name__ == "__main__":
    cams = Webcams()
    for device in cams.device_list():
        print(f'dev: {str(device)}')


#mport the libraries
import cv2 as cv
import numpy as np
import time
import sys
import yaml
from FPS import FPS
from threading import Thread, Lock

class ProjectedTarget:
    def __init__(self, camNum) -> None:
        self.threshold = 5000
        self.camNum = camNum
        #create a mask for green colour using inRange function
        #read the image
        self.calibration = {}
        self.calibration["imgzoom"] = 1
        self.calibration["hstretch"] = 1
        self.calibration["xshift"] = 0
        self.calibration["yshift"] = 0
        self.calibration["hit_radius"] = 5
        self.hits = []
        self.thits = []
        self.height = 0
        self.length = 0
        self.center = None

        #set the lower and upper bounds for the green hue
        self.lower_red = np.array([0,95,95])
        self.upper_red = np.array([10,255,255])
        self.running = True
        self.lastHitTime = time.time()
        self.hitslock = Lock()
    def capture(self):
        print("Capture started")
        lastHitTime = time.time()
        fps = FPS().start()
        self.cap = cv.VideoCapture(self.camNum)
        while self.running:
            fps.update()
            ret, img = self.cap.read()
            #Do all processing here
            self.process(img) 
        self.cap.release()
        print(f"FPS: {fps.fps()}")

    def start_calibration(self):
        multiplier = 1
        self.cap = cv.VideoCapture(self.camNum)
        #call at least once before starting loop to ensure we have a valid image before threading
        self.load_calibration() 
        while True:
            ret, img = self.cap.read()
            img = self.reshape(img) 
            dimen = img.shape
            self.height = dimen[0]
            self.length = dimen[1]
            self.center = ( int(self.length / 2), int(self.height / 2))
            self.radius = dimen[0] / 12
            cal = np.zeros((self.height,self.length, 3), dtype=np.uint8)
            cal.fill(255)
        
            #convert the BGR image to HSV colour space
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, self.lower_red, self.upper_red)
            masked = cv.bitwise_and(cal, cal, mask=mask)
        
            #Create new projected image
            for i in range(6):
                cv.circle(masked, self.center , int(i * self.radius), (0, 0, 255), 2, cv.LINE_AA)

            #draw sample hit at bullseye 
            cv.circle(masked, self.center, self.calibration["hit_radius"], (0, 255, 0), 2, cv.LINE_AA)
            cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
            cv.imshow("Calibration",masked)
            cv.imshow("Calibration",masked)
            
            keyOrd = cv.waitKey(1)
            if keyOrd > 0:
                key = chr(keyOrd)
                if key == 'c':
                    break
                if key == '+':
                    self.calibration["imgzoom"] = self.calibration["imgzoom"] + (0.01 * multiplier)
                    multiplier = 1
                if key == '-':
                    self.calibration["imgzoom"] = self.calibration["imgzoom"] - (0.01 * multiplier)
                    multiplier = 1
                if key == '3':
                    self.calibration["hstretch"] = self.calibration["hstretch"]+ (0.01 * multiplier)
                    multiplier = 1
                if key == '1':
                    self.calibration["hstretch"] = self.calibration["hstretch"]- (0.01 * multiplier)
                    multiplier = 1
                if key == '8':
                    self.calibration["yshift"] = self.calibration["yshift"] + (1 * multiplier)
                    multiplier = 1
                if key == '2':
                    self.calibration["yshift"] = self.calibration["yshift"] - (1 * multiplier)
                    multiplier = 1
                if key == '6':
                    self.calibration["xshift"] = self.calibration["xshift"] + (1 * multiplier)
                    multiplier = 1
                if key == '4':
                    self.calibration["xshift"] = self.calibration["xshift"] - (1 * multiplier)
                    multiplier = 1
                if key == '9':
                    self.calibration["hit_radius"] = self.calibration["hit_radius"] + (1 * multiplier)
                    multiplier = 1
                if key == '7':
                    self.calibration["hit_radius"] = self.calibration["hit_radius"] - (1 * multiplier)
                    multiplier = 1
                if key == '*':
                    multiplier = 10
                if key == 's':
                    self.save_calibration()
                if key == 'l':
                    self.load_calibration()
                if key == 'Y':
                    self.upper_red[0] = self.upper_red[0] + 1
                if key == 'y':
                    self.upper_red[0] = self.upper_red[0] - 1
                if key == 'U':
                    self.upper_red[1] = self.upper_red[1] + 1
                if key == 'u':
                    self.upper_red[1] = self.upper_red[1] - 1
                if key == 'I':
                    self.upper_red[2] = self.upper_red[2] + 1
                if key == 'i':
                    self.upper_red[2] = self.upper_red[2] - 1
                if key == 'H':
                    self.lower_red[0] = self.lower_red[0] + 1
                if key == 'h':
                    self.lower_red[0] = self.lower_red[0] - 1
                if key == 'J':
                    self.lower_red[1] = self.lower_red[1] + 1
                if key == 'j':
                    self.lower_red[1] = self.lower_red[1] - 1
                if key == 'K':
                    self.lower_red[2] = self.lower_red[2] + 1
                if key == 'k':
                    self.lower_red[2] = self.lower_red[2] - 1
        cv.destroyAllWindows()
        time.sleep(1)
        print("calibration destroyed?")
        
    def find_center(self, img):
    
        # convert the image to grayscale
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     
        # convert the grayscale image to binary image
        ret,thresh = cv.threshold(gray_image,127,255,0)
     
        # find moments in the binary image
        # im2, contours, hierarchy = cv.findContours(array,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        M = cv.moments(thresh)
    
        # calculate x,y coordinate of center
        if M["m00"] > self.threshold:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY) 
        else:
            return None
    def process(self, frame):
        if not self.running:
            print("Not running, abort processing")
            return
        # Take each frame
        frame = self.reshape(frame) 
        dimen = frame.shape
        # get dimensions
        self.height = dimen[0]
        self.length = dimen[1]
        self.center = ( int(self.length / 2), int(self.height / 2))
        self.radius = dimen[0] / 12
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_red, self.upper_red)
        masked = cv.bitwise_and(frame, frame, mask=mask)
        maxLoc = self.find_center(masked)
        if maxLoc is not None:
            if ( (time.time() - self.lastHitTime) > 1 ):
                self.hitslock.acquire()
                if (maxLoc not in self.hits):
                    self.hits.append(maxLoc)
                    print(maxLoc)
                    self.lastHitTime = time.time()
                self.hitslock.release()

    def start(self):
        # Start image capture
        captureThread = Thread(target=self.capture, args=())
        captureThread.start()
        print("Starting...")
        while True:
            proj = np.zeros((self.height, self.length,3), dtype=np.uint8)

            #Draw Target 
            for i in range(6):
                cv.circle(proj, self.center , int(i * self.radius), (255, 255, 255), 2, cv.LINE_AA)
            #Draw Hits 
            self.hitslock.acquire()
            for hit in self.hits:
                if hit != (0, 0):
                    cv.circle(proj, hit, self.calibration["hit_radius"], (0, 255, 0), 2, cv.LINE_AA)
            self.hitslock.release()
    
    
            cv.namedWindow("Target", cv.WINDOW_NORMAL)
            cv.imshow('Target', proj)
    
            keyOrd = cv.waitKey(1000)
            key = ''
            if keyOrd > 0:
                key = chr(keyOrd)
    
            if key == 'q':
                self.running = False
                break
            if key == 'c':
                self.hits.clear()
                update = True
                print("hits cleared")
            if key == '+':
                threshold = threshold - 10
                print("Sensitivity increased")
            if key == '-':
                threshold = threshold + 10
                print("Sensitivity decreased")
        cv.destroyAllWindows()
    
    def apply_zoom(self, img, angle=0, coord=None):
        cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    
        rot_mat = cv.getRotationMatrix2D((cx,cy), angle, self.calibration["imgzoom"])
        result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
        return result
    
    def reshape(self, frame):
        if not self.running:
            return
        img = self.apply_zoom(frame)
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        newWidth = int(imgWidth  * self.calibration["hstretch"])
        newImg = cv.resize(img, (newWidth,imgHeight), interpolation = cv.INTER_AREA)
        # if stretching, need to crop
        if self.calibration["hstretch"] > 1:
            start = int((newWidth - imgWidth) / 2)
            stop = int(newWidth - start)
            newImg = newImg[0:imgHeight, start:stop]#, 0:imgHeight]
        if self.calibration["hstretch"] < 1:
            padding = int((imgWidth - newWidth) / 2)
            newImg = cv.copyMakeBorder(newImg, 0, 0, padding, padding, cv.BORDER_CONSTANT)
        if self.calibration["xshift"] > 0:
            newImg = cv.copyMakeBorder(newImg, 0, 0, abs(self.calibration["xshift"]), 0, cv.BORDER_CONSTANT)
            newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]
        if self.calibration["xshift"] < 0:
            newImg = cv.copyMakeBorder(newImg, 0, 0, 0, abs(self.calibration["xshift"]), cv.BORDER_CONSTANT)
            newImg = newImg[0:imgHeight, abs(self.calibration["xshift"]):imgWidth+abs(self.calibration["xshift"])]#, 0:imgHeight]
        if self.calibration["yshift"] > 0:
            newImg = cv.copyMakeBorder(newImg, 0, abs(self.calibration["yshift"]), 0, 0, cv.BORDER_CONSTANT)
            newImg = newImg[abs(self.calibration["yshift"]):imgHeight+abs(self.calibration["yshift"]), 0:imgWidth]#, 0:imgHeight]
        if self.calibration["yshift"] < 0:
            newImg = cv.copyMakeBorder(newImg, abs(self.calibration["yshift"]), 0, 0, 0, cv.BORDER_CONSTANT)
            newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]
        return newImg
    
    def save_calibration(self):
        with open("calibration_data.yaml", 'w') as caldata:
            yaml.dump(self.calibration, caldata)

    def load_calibration(self):
        if os.path.exists("calibration_data.yaml"):
            with open("calibration_data.yaml", 'r') as caldata:
                self.calibration = yaml.safe_load(caldata)
            print("Loaded calibration data")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        camNum = sys.argv[1]
    else:
        camNum = 0
    print(f"Cam Num: {camNum}")
    target = ProjectedTarget(camNum)
    target.start_calibration()    
    time.sleep(1) 
    print("cal complete, slept, starting") 
    target.start()
            
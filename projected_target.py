
#mport the libraries
import cv2 as cv
import numpy as np
import time
import sys
import yaml

THRESHOLD = 1000
#create a mask for green colour using inRange function

#read the image
calibration = {}
calibration["imgzoom"] = 1
calibration["hstretch"] = 1
calibration["xshift"] = 0
calibration["yshift"] = 0
calibration["hit_radius"] = 5

camNum = int(sys.argv[1])
cap = cv.VideoCapture(camNum)
hits = []
thits = []
#set the lower and upper bounds for the green hue
lower_red = np.array([0,95,95])
upper_red = np.array([10,255,255])

def start_calibration():
    multiplier = 1
    while True:
        global calibration
        ret, img = cap.read()
        img = apply_zoom (img) 
        img = reshape (img) 
        dimen = img.shape
        height = dimen[0]
        length = dimen[1]
        center = ( int(length / 2), int(height / 2))
        radius = dimen[0] / 12
    
        cal = np.zeros((height,length, 3), dtype=np.uint8)
        cal.fill(255)
    
        #convert the BGR image to HSV colour space
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
        mask = cv.inRange(hsv, lower_red, upper_red)
    
        masked = cv.bitwise_and(cal, cal, mask=mask)
    
        #Create new projected image
    
        for i in range(6):
            cv.circle(masked, center , int(i * radius), (0, 0, 255), 2, cv.LINE_AA)
        #draw sample hit at bullseye 
        cv.circle(masked, center, calibration["hit_radius"], (0, 255, 0), 2, cv.LINE_AA)
        cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    
#        cv.setWindowProperty("Calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Calibration",masked)
        keyOrd = cv.waitKey(1)
        if keyOrd > 0:
            key = chr(keyOrd)
            if key == 'c':
                break
            if key == '+':
                calibration["imgzoom"] = calibration["imgzoom"] + (0.01 * multiplier)
                multiplier = 1
            if key == '-':
                calibration["imgzoom"] = calibration["imgzoom"] - (0.01 * multiplier)
                multiplier = 1
            if key == '3':
                calibration["hstretch"] = calibration["hstretch"]+ (0.01 * multiplier)
                multiplier = 1
            if key == '1':
                calibration["hstretch"] = calibration["hstretch"]- (0.01 * multiplier)
                multiplier = 1
            if key == '8':
                calibration["yshift"] = calibration["yshift"] + (1 * multiplier)
                multiplier = 1
            if key == '2':
                calibration["yshift"] = calibration["yshift"] - (1 * multiplier)
                multiplier = 1
            if key == '6':
                calibration["xshift"] = calibration["xshift"] + (1 * multiplier)
                multiplier = 1
            if key == '4':
                calibration["xshift"] = calibration["xshift"] - (1 * multiplier)
                multiplier = 1
            if key == '9':
                calibration["hit_radius"] = calibration["hit_radius"] + (1 * multiplier)
                multiplier = 1
            if key == '7':
                calibration["hit_radius"] = calibration["hit_radius"] - (1 * multiplier)
                multiplier = 1
            if key == '*':
                multiplier = 10
            if key == 's':
                save_calibration()
            if key == 'l':
                load_calibration()
            if key == 'Y':
                upper_red[0] = upper_red[0] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'y':
                upper_red[0] = upper_red[0] - 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'U':
                upper_red[1] = upper_red[1] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'u':
                upper_red[1] = upper_red[1] - 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'I':
                upper_red[2] = upper_red[2] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'i':
                upper_red[2] = upper_red[2] - 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'H':
                lower_red[0] = lower_red[0] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'h':
                lower_red[0] = lower_red[0] - 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'J':
                lower_red[1] = lower_red[1] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'j':
                lower_red[1] = lower_red[1] - 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'K':
                lower_red[2] = lower_red[2] + 1
                print(f'{upper_red}')
                print(f'{lower_red}')
            if key == 'k':
                lower_red[2] = lower_red[2] - 1
                print(f'{upper_red}')

                
   
    cv.destroyAllWindows()
    
def find_center(img):

    # convert the image to grayscale
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray_image,127,255,0)
   
    cv.destroyAllWindows()
    
def find_center(img):

    # convert the image to grayscale
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray_image,127,255,0)
 
    # find moments in the binary image
    # im2, contours, hierarchy = cv.findContours(array,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    M = cv.moments(thresh)

    # calculate x,y coordinate of center
    if M["m00"] > THRESHOLD:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY) 
    else:
        return None
    
def start_target():
    lastHitTime = time.time()
    global calibration
    while True:

        # Take each frame
        ret, frame = cap.read()
        frame = apply_zoom (frame) 
        frame = reshape(frame)

        # get dimensions
        dimen = frame.shape
        height = dimen[0]
        length = dimen[1]
        center = ( int(length / 2), int(height / 2))
        radius = dimen[0] / 12

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, lower_red, upper_red)
    
        masked = cv.bitwise_and(frame, frame, mask=mask)
#        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(mask)
        maxLoc = find_center(masked)
        if maxLoc is not None:

           if ( (time.time() - lastHitTime) > 1 ) and (maxLoc not in hits):
                hits.append(maxLoc)
                print(maxLoc)
                lastHitTime = time.time()
       
        proj = np.zeros((height,length,3), dtype=np.uint8)
        #Draw Target 
        for i in range(6):
            cv.circle(proj, center , int(i * radius), (255, 255, 255), 2, cv.LINE_AA)
        #Draw Hits 
        for hit in hits:
            if hit != (0, 0):
                cv.circle(proj, hit, calibration["hit_radius"], (0, 255, 0), 2, cv.LINE_AA)

        cv.namedWindow("Target", cv.WINDOW_NORMAL)
       # cv.setWindowProperty("Target", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Target', proj)

        keyOrd = cv.waitKey(1)
        key = ''
        if keyOrd > 0:
            key = chr(keyOrd)

        if key == 'q':
            break
        if key == 'c':
            hits.clear()
            print("hits cleared")

    cap.release()
    cv.destroyAllWindows()

def apply_zoom(img, angle=0, coord=None):
    global calibration
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]

    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, calibration["imgzoom"])
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def reshape(img):
    
    global calibration
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    newWidth = int(imgWidth  * calibration["hstretch"])
    newImg = cv.resize(img, (newWidth,imgHeight), interpolation = cv.INTER_AREA)
    # if stretching, need to crop
    if calibration["hstretch"] > 1:
        start = int((newWidth - imgWidth) / 2)
        stop = int(newWidth - start)
        newImg = newImg[0:imgHeight, start:stop]#, 0:imgHeight]
    if calibration["hstretch"] < 1:
        padding = int((imgWidth - newWidth) / 2)
        newImg = cv.copyMakeBorder(newImg, 0, 0, padding, padding, cv.BORDER_CONSTANT)
    if calibration["xshift"] > 0:
        newImg = cv.copyMakeBorder(newImg, 0, 0, abs(calibration["xshift"]), 0, cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]
    if calibration["xshift"] < 0:
        newImg = cv.copyMakeBorder(newImg, 0, 0, 0, abs(calibration["xshift"]), cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, abs(calibration["xshift"]):imgWidth+abs(calibration["xshift"])]#, 0:imgHeight]
    if calibration["yshift"] > 0:
        newImg = cv.copyMakeBorder(newImg, 0, abs(calibration["yshift"]), 0, 0, cv.BORDER_CONSTANT)
        newImg = newImg[abs(calibration["yshift"]):imgHeight+abs(calibration["yshift"]), 0:imgWidth]#, 0:imgHeight]
    if calibration["yshift"] < 0:
        newImg = cv.copyMakeBorder(newImg, abs(calibration["yshift"]), 0, 0, 0, cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]

    return newImg
def save_calibration():
    global calibration
    with open("calibration_data.yaml", 'w') as caldata:
        yaml.dump(calibration, caldata)
def load_calibration():
    global calibration
    with open("calibration_data.yaml", 'r') as caldata:
        calibration = yaml.safe_load(caldata)
    print("Loaded calibration data")

start_calibration()
start_target()

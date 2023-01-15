
#mport the libraries
import cv2 as cv
import numpy as np
import time
import sys

THRESHOLD = 1000
#create a mask for green colour using inRange function

#read the image
imgzoom = 1
camNum = int(sys.argv[1])
cap = cv.VideoCapture(camNum)
hits = []
#set the lower and upper bounds for the green hue
lower_red = np.array([0,95,95])
upper_red = np.array([10,255,255])

def start_calibration():
    while True:
        global imgzoom
        ret, img = cap.read()
        img = zoom_at (img, imgzoom) 
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
    
        cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    
#        cv.setWindowProperty("Calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Calibration",masked)
        key = cv.waitKey(1)
        
        if key == ord('c'):
            break
        if key == ord('='):
            print(f'zoom in: {imgzoom}')
            imgzoom = imgzoom + 0.01
        if key == ord('-'):
            print(f'zoom out: {imgzoom}')
            imgzoom = imgzoom - 0.01
        if key == ord('+'):
            print(f'zoom in: {imgzoom}')
            imgzoom = imgzoom + 0.1
        if key == ord('_'):
            print(f'zoom out: {imgzoom}')
            imgzoom = imgzoom - 0.1
    
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
    global imgzoom
    while True:

        # Take each frame
        ret, frame = cap.read()
        frame = zoom_at (frame, imgzoom) 
 
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


            print(f'center of mass: {maxLoc}')         
            if ( (time.time() - lastHitTime) > 1 ) and (maxLoc not in hits):
                print(f"New Hit {maxLoc}")

                hits.append(maxLoc)
                lastHitTime = time.time()
       
        proj = np.zeros((height,length,3), dtype=np.uint8)
        #Draw Target 
        for i in range(6):
            cv.circle(proj, center , int(i * radius), (255, 255, 255), 2, cv.LINE_AA)
        #Draw Hits 
        for hit in hits:
            if hit != (0, 0):
                cv.circle(proj, hit, 20, (0, 255, 0), 2, cv.LINE_AA)

        cv.namedWindow("Target", cv.WINDOW_NORMAL)
       # cv.setWindowProperty("Target", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Target', proj)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            hits.clear()
            print("hits cleared")

    cap.release()
    cv.destroyAllWindows()

def zoom_at(img, zoom=1, angle=0, coord=None):

    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]

    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

start_calibration()
start_target()

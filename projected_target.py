
#mport the libraries
import cv2 as cv
import numpy as np
import time


THRESHOLD = 1000
#create a mask for green colour using inRange function

#read the image
cap = cv.VideoCapture(0)
hits = []

#set the lower and upper bounds for the green hue
lower_red = np.array([0,95,95])
upper_red = np.array([10,255,255])

def start_calibration():
    while True:
        ret, img = cap.read()
    
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
    
        cv.setWindowProperty("Calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Calibration",masked)
        if cv.waitKey(1) & 0xFF == ord('c'):
            break
    
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
    while True:

        # Take each frame
        ret, frame = cap.read()
 
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
        cv.setWindowProperty("Target", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow('Target', proj)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('c'):
            hits.clear()
            print("hits cleared")

    cap.release()
    cv.destroyAllWindows()

start_calibration()
start_target()

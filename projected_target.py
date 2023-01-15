
#mport the libraries
import cv2 as cv
import numpy as np
import time
import sys

THRESHOLD = 1000
#create a mask for green colour using inRange function

#read the image
imgzoom = 1
vstretch = 1
hstretch = 1
xshift = 0
yshift = 0
camNum = int(sys.argv[1])
cap = cv.VideoCapture(camNum)
hits = []
#set the lower and upper bounds for the green hue
lower_red = np.array([0,95,95])
upper_red = np.array([10,255,255])

def start_calibration():
    multiplier = 1
    while True:
        global imgzoom
        global hstretch
        global vstretch
        global xshift
        global yshift
        ret, img = cap.read()
        img = zoom_at (img, imgzoom) 
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
    
        cv.namedWindow("Calibration", cv.WINDOW_NORMAL)
    
#        cv.setWindowProperty("Calibration", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Calibration",masked)
        keyOrd = cv.waitKey(1)
        if keyOrd > 0:
            key = chr(keyOrd)
            if key == 'c':
                break
            if key == '+':
                imgzoom = imgzoom + (0.01 * multiplier)
                multiplier = 1
            if key == '-':
                imgzoom = imgzoom - (0.01 * multiplier)
                multiplier = 1
            if key == '3':
                hstretch = hstretch + (0.01 * multiplier)
                multiplier = 1
            if key == '1':
                hstretch = hstretch - (0.01 * multiplier)
                multiplier = 1
            if key == '8':
                yshift = yshift + (1 * multiplier)
                multiplier = 1
            if key == '2':
                yshift = yshift - (1 * multiplier)
                multiplier = 1
            if key == '6':
                xshift = xshift + (1 * multiplier)
                multiplier = 1
            if key == '4':
                xshift = xshift - (1 * multiplier)
                multiplier = 1
            if key == '*':
                multiplier = 10
   
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
    global hstretch
    global vstretch
    global xshift
    global yshift
    while True:

        # Take each frame
        ret, frame = cap.read()
        frame = zoom_at (frame, imgzoom) 
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

def zoom_at(img, zoom=1, angle=0, coord=None):
    global imgzoom
    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]

    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, imgzoom)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
def reshape(img):
    
    global hstretch
    global xshift
    global yshift
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    newWidth = int(imgWidth  * hstretch)
    newImg = cv.resize(img, (newWidth,imgHeight), interpolation = cv.INTER_AREA)
    # if stretching, need to crop
    if hstretch > 1:
        start = int((newWidth - imgWidth) / 2)
        stop = int(newWidth - start)
        newImg = newImg[0:imgHeight, start:stop]#, 0:imgHeight]
    if hstretch < 1:
        padding = int((imgWidth - newWidth) / 2)
        newImg = cv.copyMakeBorder(newImg, 0, 0, padding, padding, cv.BORDER_CONSTANT)
    if xshift > 0:
        newImg = cv.copyMakeBorder(newImg, 0, 0, abs(xshift), 0, cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]
    if xshift < 0:
        newImg = cv.copyMakeBorder(newImg, 0, 0, 0, abs(xshift), cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, abs(xshift):imgWidth+abs(xshift)]#, 0:imgHeight]
    if yshift > 0:
        newImg = cv.copyMakeBorder(newImg, 0, abs(yshift), 0, 0, cv.BORDER_CONSTANT)
        newImg = newImg[abs(yshift):imgHeight+abs(yshift), 0:imgWidth]#, 0:imgHeight]
    if yshift < 0:
        newImg = cv.copyMakeBorder(newImg, abs(yshift), 0, 0, 0, cv.BORDER_CONSTANT)
        newImg = newImg[0:imgHeight, 0:imgWidth]#, 0:imgHeight]

    return newImg
start_calibration()
start_target()

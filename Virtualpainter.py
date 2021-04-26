import cv2
import os 
import handTrackingModule as htm
import numpy as np
import time


################
drwaColour = (255,0,255)
folderPath = "Header"
brushThickness = 8
eraserThickness = 40
################

myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    images = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(images)
# print(len(overlayList))
header = overlayList[0]
###############

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(maxHands=1,detectionCon=0.40)
xp,yp = 0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)
##############################

while True:
    ## Import Image
    success,img = cap.read()
    img = cv2.flip(img,1)

    ## Find Hand landmarlks

    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    if len(lmList)!=0:
        # print(lmList)

        # tip of index and middle fingre
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        ## Check which Fingers are up
        fingre= detector.finguresUp()
        # print(fingre)


        ## Finger Selection - Two fingers are up
        if fingre[1] and fingre[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            ## Checking for the click
            if y1 < 135:
                if 250<x1<450:
                    header = overlayList[0]
                    drwaColour = (255, 0, 255)
                elif 550<x1<750:
                    header = overlayList[1]
                    drwaColour = (255, 0, 0)
                elif 800<x1<950:
                    header = overlayList[2]
                    drwaColour = (0, 255, 0)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drwaColour = (0,0,0)
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drwaColour,cv2.FILLED)

        ## Drawing Mode - Index Finger up
        if fingre[1] and fingre[2] == False:
            cv2.circle(img,(x1,y1),15,(0,0,133),cv2.FILLED)
            print("Drawing Mode")

            if xp ==0 and yp == 0:
                xp,yp = x1,y1

            if drwaColour == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drwaColour, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drwaColour, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drwaColour,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drwaColour,brushThickness)

            xp,yp =x1,y1
    imgGrag = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGrag,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    ## Setting the Screen Header
    img[0:138,0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cv2.imshow("Images",img)
    # cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)
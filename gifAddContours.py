import sys
import os.path
import os
from PIL import Image
import cv2
import numpy as np


def showImage(windowName, image, windowX=0, windowY=0):
    cv2.namedWindow(windowName)  # Create a named window
    cv2.moveWindow(windowName, windowX, windowY)  # Move it to new location
    cv2.imshow(windowName, image)


def getImageScaleInfo(contours):
    # find scale contour, and return the width in pixels
    for cntr in contours:
        (ulx, uly, wid, hgt) = cv2.boundingRect(cntr)
        # exclude contours that are really wide or really tall relatively
        if ulx > 450 and uly > 550 and wid / hgt > 15:
            # found the scale contour
            # print("found", wid)
            return wid

    # print("not found")
    return None


def addContour(img):
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("gray image: ", imgGray.shape)

    # create mask to remove the 'scale' and legend '10 um'
    excludeRegion = np.ones(imgGray.shape[:2], dtype=np.uint8)*255
    # cv2.rectangle(excludeRegion, (531, 634), (627, 637), 0, -1)
    cv2.rectangle(excludeRegion, (560, 643), (597, 655), 0, -1)

    # exclude the bottom 2 rows
    cv2.rectangle(excludeRegion, (0, imgHeight-2), (imgWidth, imgHeight-1), 0, -1)
    # print(excludeRegion)

    threshold, threshImage = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold, threshImage = cv2.threshold(imgGray, threshold + 40, 255, cv2.THRESH_BINARY)

    imgWindowYOffset = 100
    threshImage = cv2.bitwise_and(threshImage, excludeRegion)

    # find contours
    contours, hierarch = cv2.findContours(threshImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    imgWithContourHull = img.copy()

    # sort contours by area size, the largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    scaleInfo = getImageScaleInfo(contours)     # how many pixels in 10 micrometer
    if scaleInfo is not None:
        print("scale, 10um is %d pixels" % scaleInfo)
        pixelSize = 10.0 / scaleInfo                # in micrometer
    else:
        print("scale not found")
        pixelSize = 10.0 / 96

    minArea = 16
    count = 0
    totalArea = 0.0
    for cntr in contours:
        (ulx, uly, wid, hgt) = cv2.boundingRect(cntr)
        # exclude contours that are really wide or really tall relatively
        if wid/hgt > 20 or hgt/wid > 20:
            # skip this one
            continue

        area = cv2.contourArea(cntr)
        areaPhysicalSize = area * pixelSize
        print("contour area: %d pixels %f um" % (area, areaPhysicalSize))
        if area < minArea:
            # skip the rest, all small areas
            continue

        # print(count, area, wid/hgt)
        count += 1
        totalArea += area

        convHull = cv2.convexHull(cntr)
        cv2.drawContours(imgWithContourHull, [convHull], -1, (255, 255, 0), 2)
        (centerX, centerY), radius = cv2.minEnclosingCircle(cntr)
        centerX = int(centerX)
        centerY = int(centerY)
        text = "%d" % count
        (textWidth, textHeight), textBaseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        cv2.putText(imgWithContourHull, text, (centerX-textWidth//2, centerY+textHeight//2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        # draw the largest contours
        # if count > 50:
        #     break
    return imgWithContourHull


def gifAddContours(gifFilePath):
    gifImage = Image.open(gifFilePath)
    framesWithContour = []
    for frameNo in range(gifImage.n_frames):
        print(frameNo)
        tempFrameFileName = "temp.png"
        gifImage.seek(frameNo)
        gifImage.save(tempFrameFileName)
        img = cv2.imread(tempFrameFileName)
        imgWithContourHull = addContour(img)
        imgWithContourHull = cv2.cvtColor(imgWithContourHull, cv2.COLOR_BGR2RGB)
        imgWithContourHull = Image.fromarray(imgWithContourHull)
        framesWithContour.append(imgWithContourHull)

    # save all frames as gif
    framesWithContour[0].save("result.gif", save_all=True, optimize=False,
                              append_images=framesWithContour[1:],
                              duration=gifImage.info['duration'],
                              loop=gifImage.info['loop'])


def main():
    gifAddContours('Sample_K2_RT_60x_1_MMStack_frame_00_50.gif')


if __name__ == "__main__":
    main()

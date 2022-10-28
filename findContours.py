import os.path
import cv2
import numpy as np


def showImage(windowName, image, windowX=0, windowY=0):
    cv2.namedWindow(windowName)  # Create a named window
    cv2.moveWindow(windowName, windowX, windowY)  # Move it to new location
    cv2.imshow(windowName, image)

def getImageScaleInfo(contours):
    # find scale contour, and return the width in pixels
    foundWidth = []
    for cntr in contours:
        (ulx, uly, wid, hgt) = cv2.boundingRect(cntr)
        # print(ulx, uly, wid, hgt)
        # exclude contours that are really wide or really tall relatively
        if ulx > 450 and uly > 550 and wid / hgt > 15:
            # found the scale contour
            # print("found", wid)
            foundWidth.append(wid)

    if len(foundWidth) == 1:
        return foundWidth[0]
    else:
        # print("not found")
        return None


imgFile = 'Sample_K2_RT_60x_1_MMStack/Sample_K2_RT_60x_1_MMStack_frame_000.png'
# imgFile = 'Sample_K2_RT_60x_1_MMStack/Sample_K2_RT_60x_1_MMStack_frame_964.png'
img = cv2.imread(imgFile)
showImage("original", img, 0, 0)

imgHeight = img.shape[0]
imgWidth = img.shape[1]
# print(img.shape)
# print(imgHeight, imgWidth)

# imgGray = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("gray image: ", imgGray.shape)

# create mask to remove the 'scale' and legend '10 um'
excludeRegion = np.ones(imgGray.shape[:2], dtype=np.uint8)*255
cv2.rectangle(excludeRegion, (531, 634), (627, 637), 0, -1)
cv2.rectangle(excludeRegion, (560, 643), (597, 655), 0, -1)

# exclude the bottom 2 rows
cv2.rectangle(excludeRegion, (0, imgHeight-2), (imgWidth, imgHeight-1), 0, -1)
# print(excludeRegion)

# threshold, threshImage = cv2.threshold(imgGray, 200, 255, cv2.THRESH_BINARY)
threshold, threshImage = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
threshold, threshImage = cv2.threshold(imgGray, threshold+40, 255, cv2.THRESH_BINARY)
# print("threshold:", threshold)
# threshImage = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)
# threshImage = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
showImage("thresh", threshImage, imgWidth, 0)

imgWindowYOffset = 100
threshImage = cv2.bitwise_and(threshImage, excludeRegion)
showImage("thresh exclude region", threshImage, imgWidth, imgHeight + imgWindowYOffset)
showImage("exclude region", excludeRegion, 0, imgHeight + imgWindowYOffset)

# find contours
contours, hierarch = cv2.findContours(threshImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# draw contours in copy of color images
imgWithContours = img.copy()
cv2.drawContours(imgWithContours, contours, -1, (0, 0, 255), 2)
showImage("Original with Contour", imgWithContours, imgWidth * 2, 0)

# try other contour shape
imgWithContourRect = img.copy()
imgWithContourHull = img.copy()

# sort contours by area size, largest first
contours = sorted(contours, key=cv2.contourArea, reverse=True)
imageScaleInfo = getImageScaleInfo(contours)
if imageScaleInfo is not None:
    print("scale, 10um is %d pixels" % imageScaleInfo)
else:
    print("image scale not found")

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
    if area < minArea:
        # skip the rest, all small areas
        continue

    print(count, area, wid/hgt)
    count += 1
    totalArea += area
    print("contour:", count, (ulx, uly, wid, hgt), wid/hgt)

    cv2.rectangle(imgWithContourRect, (ulx, uly), (ulx + wid, uly + hgt), (0, 0, 255), 2)

    convHull = cv2.convexHull(cntr)
    cv2.drawContours(imgWithContourHull, [convHull], -1, (255, 255, 0), 2)
    (centerX, centerY), radius = cv2.minEnclosingCircle(cntr)
    centerX = int(centerX)
    centerY = int(centerY)
    text = "%d" % count
    (textWidth, textHeight), textBaseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    print(text, "textSize", textWidth, textHeight, textBaseline)
    cv2.putText(imgWithContourHull, text, (centerX-textWidth//2, centerY+textHeight//2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

    showImage("Original with Contour Rectangle", imgWithContourRect, imgWidth * 3, 0)
    showImage("Original with Contour Hull", imgWithContourHull, imgWidth * 4, 0)

    # draw the largest contours
    # if count > 20:
    #     break
    # enable the following to show one contour with each keypress
    # ch = chr(0xFF & cv2.waitKey())
    # if ch == 'q':
    #     break

# save image with contour
imgFileName, imgFileExt = os.path.splitext(imgFile)
cv2.imwrite(imgFileName+"_contours.png", imgWithContourHull)

print("total contours:", len(contours))
print("shown contours:", count)
print("shown contour area:", totalArea)

while True:
    ch = chr(0xFF & cv2.waitKey())
    if ch == 'q':
        break
cv2.destroyAllWindows()

import numpy as np
import cv2
from PIL import Image
# import argparse
#
# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# args = parser.parse_args(parse_args)


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

    scaleInfo = getImageScaleInfo(contours)
    if scaleInfo is not None:
        print("scale, 10um is %d pixels" % scaleInfo)
    else:
        print("scale not found")

    minArea = 10
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
        if count > 50:
            break
    return imgWithContourHull


def getContourImg(img, maxContours=20):
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("gray image: ", imgGray.shape)

    # create mask to remove the 'scale' and legend '10 um'
    excludeRegion = np.ones(imgGray.shape[:2], dtype=np.uint8)*255
    cv2.rectangle(excludeRegion, (531, 634), (627, 637), 0, -1)
    cv2.rectangle(excludeRegion, (560, 643), (597, 655), 0, -1)

    # exclude the bottom 2 rows
    cv2.rectangle(excludeRegion, (0, imgHeight-2), (imgWidth, imgHeight-1), 0, -1)
    # print(excludeRegion)

    threshold, threshImage = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshold, threshImage = cv2.threshold(imgGray, threshold + 40, 255, cv2.THRESH_BINARY)

    threshImage = cv2.bitwise_and(threshImage, excludeRegion)

    # find contours
    contours, hierarch = cv2.findContours(threshImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    imgWithContour = np.zeros_like(img)

    # sort contours by area size, the largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

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

        # print(count, area, wid/hgt)
        count += 1
        totalArea += area

        cv2.drawContours(imgWithContour, [cntr], -1, (255, 255, 255), 1)

        # draw the largest contours
        # if count >= maxContours:
        #     break

    return imgWithContour


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


gifImage = Image.open("Sample_K2_RT_60x_1_MMStack.gif")
tempFrameFileName = "frame.png"
gifImage.seek(0)    # 1st frame
gifImage.save(tempFrameFileName)
old_frame = cv2.imread(tempFrameFileName)
imgHeight = old_frame.shape[0]
imgWidth = old_frame.shape[1]
windowOffsetX = imgWidth + 10
windowOffsetY = imgHeight + 40

maxCoutourCount = 20
# old_frame = addContour(old_frame)
old_frame = getContourImg(old_frame, maxCoutourCount)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

framesWithOpticalFlow = []
for frameNo in range(1, gifImage.n_frames, 20):
    print(frameNo)
    gifImage.seek(frameNo)
    gifImage.save(tempFrameFileName)
    original = cv2.imread(tempFrameFileName)
    showImage('Original', original)
    # frame = addContour(frame)
    contour_frame = getContourImg(original, maxCoutourCount)
    frame = contour_frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    showImage('Contour', contour_frame, windowOffsetX, 0)
    showImage('Contour Optical Flow', img, windowOffsetX*2, 0)

    imgWithOpticalFlow = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgWithOpticalFlow = Image.fromarray(imgWithOpticalFlow)
    framesWithOpticalFlow.append(imgWithOpticalFlow)

    k = cv2.waitKey(30) & 0xff
    if chr(k) == 'q':
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

framesWithOpticalFlow[0].save("contour_optical_flow_result.gif", save_all=True, optimize=False,
                              append_images=framesWithOpticalFlow[1:],
                              duration=gifImage.info['duration'],
                              loop=gifImage.info['loop'])
cv2.destroyAllWindows()

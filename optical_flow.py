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

gifImage = Image.open("Sample_K2_RT_60x_1_MMStack.gif")
tempFrameFileName = "frame.png"
gifImage.seek(0)    # 1st frame
gifImage.save(tempFrameFileName)
old_frame = cv2.imread(tempFrameFileName)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
imgHeight = old_frame.shape[0]
imgWidth = old_frame.shape[1]
windowOffsetX = imgWidth + 10
windowOffsetY = imgHeight + 40

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.005,
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
    frame = cv2.imread(tempFrameFileName)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    showImage('Original', frame)

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

    showImage('Optical Flow', img, windowOffsetX)

    imgWithOpticalFlow = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgWithOpticalFlow = Image.fromarray(imgWithOpticalFlow)
    framesWithOpticalFlow.append(imgWithOpticalFlow)

    k = cv2.waitKey(30) & 0xff
    if chr(k) == 'q':
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

framesWithOpticalFlow[0].save("optical_result.gif", save_all=True, optimize=False,
                              append_images=framesWithOpticalFlow[1:],
                              duration=gifImage.info['duration'],
                              loop=gifImage.info['loop'])
cv2.destroyAllWindows()

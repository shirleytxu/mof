import sys
import os
from PIL import Image


def gif2png(gifFilePath):
    gifFileDir, gifFileName = os.path.split(gifFilePath)
    gifFileNameBase, gifFileNameExt = os.path.splitext(gifFileName)

    # create directory to store individual frames
    os.mkdir(gifFileNameBase)

    gifImage = Image.open(gifFilePath)
    for frameNo in range(gifImage.n_frames):
        frameFileName = "%s_frame_%03d.png" % (gifFileNameBase, frameNo)
        print(frameFileName)
        gifImage.seek(frameNo)
        gifImage.save(os.path.join(gifFileNameBase, frameFileName))


def main():
    gif2png("Sample_K2_RT_60x_1_MMStack.gif")


if __name__ == "__main__":
    main()

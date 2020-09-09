import numpy as np
import cv2
import logging
logging.basicConfig(
    level=logging.WARN,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Binarize image
# 1. constructor stores the original image
# 2. binarize() returns a binarized version of the original image

class Binarizer:

    # Constructor - load image & prepare parameters
    def __init__(self, img):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.sx, self.sy = img.shape[1::-1]
        self.img = img
        self.log.debug("Original image dimension:"
                       " (%s, %s)" % (self.sx, self.sy))

    # Apply color threshold and gradient threshold.
    def combined(self, s_thresh=(170, 255), sx_thresh=(20, 100)):
        self.log.debug("Binarization based on color & gradient thresholds"
                       " (%s, %s) & (%s, %s)" \
                       % (*s_thresh, *sx_thresh))
        # Copy image so that the original remains the same
        img = np.copy(self.img)
        # Convert to HLS color space and separate channels
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Compute Sobel x by taking derivative in x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # Threshold x gradient
        sxbinary = np.zeros_like(sobel)
        sxbinary[(sobel >= sx_thresh[0]) & (sobel <= sx_thresh[1])] = 1
        # Threshold color channel S
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    # Binarize loaded image using selected algorithm
    def binarize(self):
        binary = self.combined()
        return binary

import logging
import numpy as np
import cv2

from lanefinder.params import binarizer_params as params

# Binarize image
# 1. constructor stores the original image
# 2. binarize() returns a binarized version of the original image

class Binarizer:

    # Constructor - load image & prepare parameters
    def __init__(self, s_thresh=None, sx_thresh=None):
        if not s_thresh:
            s_thresh = params['s_channel_color_threshold']
        if not sx_thresh:
            sx_thresh = params['x_gradient_threshold']
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.WARN)
        self.s_th, self.sx_th = s_thresh, sx_thresh
        self.log.debug("Initialize binarizer based on"
                       " color & gradient thresholds"
                       " (%s, %s) & (%s, %s)" \
                       % (*self.s_th, *self.sx_th))
        self.sx, self.sy = 0, 0
        self.img = None

    # Apply color threshold and gradient threshold.
    def combined(self):
        # Convert to HLS color space and separate channels
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Compute Sobel x by taking derivative in x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # Threshold x gradient
        sxbinary = np.zeros_like(sobel)
        sxbinary[(sobel >= self.sx_th[0]) & (sobel <= self.sx_th[1])] = 1
        # Threshold color channel S
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_th[0]) & (s_channel <= self.s_th[1])] = 1
        # Stack each channel
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    # Load & binarize image using selected algorithm
    def binarize(self, img):
        self.sx, self.sy = img.shape[1::-1]
        # Copy image so that the original (parameter) doesn't change
        self.img = np.copy(img)
        self.log.debug("Original image dimension:"
                       " (%s, %s)" % (self.sx, self.sy))
        binary = self.combined()
        return binary

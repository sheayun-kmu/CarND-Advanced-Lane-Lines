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

    # Constructor - prepare parameters
    def __init__(self, img):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.sx, self.sy = img.shape[1::-1]
        self.img = img
        self.log.debug("Original image dimension:"
                       " (%s, %s)" % (self.sx, self.sy))

    def binarize(self):
        binary = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        return binary

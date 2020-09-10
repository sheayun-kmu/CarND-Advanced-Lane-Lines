import logging
import numpy as np
import cv2

# Lane detector (image --> detected lane line)
# 1. initialize
#    - parameter initialization
# 2. histogram peak & sliding window
#    - input: binary image (thresholded bird's eye view)
#    - state change: fitted second-order polynomial (coefficients)
#    - output: fitted polynomial (coefficients)
# 3. search from previously detected lane line
#    - input: binary image (thresholded bird's eye view)
#    - state change: fitted second-order polynomial (coefficients)
#    - output: fitted polynomial (coefficients)

class LaneDetector:

    # Load parameters from configuration file.
    def __init__(self, calib_image_files=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.hor_range = (0, 0)

    # Detect a lane line by a histogram peak and sliding window
    def slide_from_peak(self, horizontal_range):
        return None

    # Detect a lane line by searching from previously found line
    def search_from_previous(self):
        return None

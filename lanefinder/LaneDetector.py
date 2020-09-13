import logging
import numpy as np
import cv2

from lanefinder.params import detector_params

# Lane detector (image --> detected lane line)
# 1. initialize
#    - parameter initialization
# 2. histogram peak & sliding window
#    - input: binary image (thresholded bird's eye view)
#    - state change: fit second-order polynomial (coefficients)
#    - output: fit polynomial (coefficients)
# 3. search from previously detected lane line
#    - input: binary image (thresholded bird's eye view)
#    - state change: fit second-order polynomial (coefficients)
#    - output: fit polynomial (coefficients)

class LaneDetector:

    # Load parameters from configuration file.
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        params = detector_params['sliding_window_params']
        self.nwindows = params['nwindows']
        self.margin = params['margin']
        self.minpix = params['minpix']
        self.log.debug("A lane detector instance has initialized.")

    # Detect a lane line by a histogram peak and sliding window.
    # Expect a two-channel (binary) image as input.
    def slide_from_peak(self, img, base_range):
        lbound, rbound = base_range
        r, c = img.shape[:2]
        # Take a histogram on the bottom half of the image.
        hist = np.sum(img[r // 2:, lbound:rbound], axis=0)
        # Let base denote the maximum peak in the range, represented
        # as an x coordinate within the image.
        base = np.argmax(hist) + lbound
        # Calculate window height
        window_height = np.int(r // self.nwindows)
        # Get x & y coordinates of all the nonzero pixels in the image.
        nonzero = img.nonzero()
        nzx, nzy = np.array(nonzero[1]), np.array(nonzero[0])
        # Initialize an empty list for collecting lane pixel indices.
        lane_pixel_inds = []
        # Step through the windows.
        delta = 0
        for w in range(self.nwindows):
            # Identify window corners.
            top = r - (w + 1) * window_height
            bottom = r - w * window_height
            left = base - self.margin
            right = base + self.margin
            # Get indices of nonzero pixels within the window.
            selected = ((top <= nzy) & (nzy <= bottom) \
                      & (left <= nzx) & (nzx <= right)).nonzero()[0]
            lane_pixel_inds.append(selected)
            # If the window contains more than minpix pixels,
            # recenter the next window by updating base.
            if len(selected) >= self.minpix:
                new_base = np.int(np.mean(nzx[selected]))
                delta = new_base - base
                base = new_base
            # Otherwise, move window in the same direction.
            else:
                base += delta

        # Flatten out the array of indices
        # (previously list of indices for each window).
        lane_pixel_inds = np.concatenate(lane_pixel_inds)
        # Let x and y contain coordinates (array) of lane line pixels.
        x, y = nzx[lane_pixel_inds], nzy[lane_pixel_inds]
        fit = np.polyfit(y, x, 2)

        return x, y, fit

    # Detect a lane line by searching from previously found line.
    # Expect a two-channel (binary) image as input.
    def search_around_prev(self, img, line):
        r, c = img.shape[:2]
        # Get x & y coordinates of all the nonzero pixels in the image.
        nonzero = img.nonzero()
        nzx, nzy = np.array(nonzero[1]), np.array(nonzero[0])
        # Collect nonzero pixels around the previously found
        # second-order polynomial with margin.
        c = line.curr_fit
        m = self.margin
        lane_pixel_inds = (
            (nzx >= c[0] * nzy ** 2 + c[1] * nzy + c[2] - m) & \
            (nzx <= c[0] * nzy ** 2 + c[1] * nzy + c[2] + m)
        )
        # Let x and y contain coordinates (array) of lane line pixels.
        x, y = nzx[lane_pixel_inds], nzy[lane_pixel_inds]
        try:
            fit = np.polyfit(y, x, 2)
        except:
            fit = (0, 0, 0)

        return x, y, fit

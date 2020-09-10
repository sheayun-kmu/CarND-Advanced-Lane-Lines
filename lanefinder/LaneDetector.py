import logging
import numpy as np
import cv2

from lanefinder.params import detector_params

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
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        params = detector_params['sliding_window_params']
        self.nwindows = params['nwindows']
        self.margin = params['margin']
        self.minpix = params['minpix']
        self.coeffs = (0, 0, 0)
        self.log.debug("A lane detector instance has initialized.")

    # Detect a lane line by a histogram peak and sliding window.
    # Expect a two-channel (binary or grayscale) image as input.
    def slide_from_peak(self, img, starting_range):
        lbound, rbound = starting_range
        r, c = img.shape[:2]
        # Initialize a copy (but 3-channel color) image
        # for visualization (debugging purposes)
        overlay = np.zeros((r, c, 3), dtype=np.uint8)
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
        for w in range(self.nwindows):
            # Identify window corners.
            top = r - (w + 1) * window_height
            bottom = r - w * window_height
            left = base - self.margin
            right = base + self.margin
            # Draw the windows (in green) for overlay visualization.
            cv2.rectangle(
                overlay,
                (left, top), (right, bottom),
                (0, 255, 0), 2
            )
            # Get indices of nonzero pixels within the window.
            selected = ((top <= nzy) & (nzy <= bottom) \
                      & (left <= nzx) & (nzx <= right)).nonzero()[0]
            lane_pixel_inds.append(selected)
            # If the window contains more than minpix pixels,
            # recenter the next window by updating base.
            if len(selected) >= self.minpix:
                base = np.int(np.mean(nzx[selected]))
        # Flatten out the array of indices
        # (previously list of indices for each window).
        lane_pixel_inds = np.concatenate(lane_pixel_inds)
        # Let x and y contain coordinates (array) of lane line pixels.
        x, y = nzx[lane_pixel_inds], nzy[lane_pixel_inds]
        # Paint pixels on lane line blue.
        overlay[y, x] = [0, 0, 255]
        self.coeffs = np.polyfit(y, x, 2)
        self.annotate_polynomial(overlay)

        return self.coeffs, overlay

    # Detect a lane line by searching from previously found line.
    def search_from_previous(self):
        return None

    # Given an image, render the fitted second order polynomial above it.
    def annotate_polynomial(self, img):
        r, c = img.shape[:2]
        f = self.coeffs
        for y in np.linspace(0, r - 1, r):
            try:
                x = f[0] * y ** 2 + f[1] * y + f[2]
            except TypeError:
                x = 1 * y ** 2 + 1 * y
            x, y = np.int(x), np.int(y)
            img[y, x - 2:x + 2] = [255, 255, 0]

    # Detect a lane line by an algorithm selected based on the
    # previous state and situation.
    def detect(self, img, starting_range):
        return self.slide_from_peak(img, starting_range)

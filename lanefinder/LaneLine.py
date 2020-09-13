import logging
import numpy as np

from lanefinder.params import detector_params as params
from lanefinder.params import conversion_params

class LaneLine:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        # whether this line has been detected previously
        self.detected = False
        # coordinates of pixels contained in lane lines (in warped space)
        self.x = None
        self.y = None
        # number of consecutive fails
        self.acc_failure = 0
        # base position (bottom) for the currently detected lane line
        self.base = 0
        # polynomial coefficients for the most recent fit
        self.curr_fit = (0, 0, 0)
        # radius of curvature (in meters)
        self.curverad = 0
        # Store last fit coefficients.
        self.prev_fits = np.empty((0, 3), dtype=np.float32)

    # Check whether this lane line has no context at all (yet).
    def very_first(self):
        return len(self.prev_fits) == 0

    # Update lane line status using a polynomial fit for the
    # newly detected lane line.
    def update(self, img_size, x, y, fit, detected):
        r, c = img_size
        self.x, self.y = x, y
        self.curr_fit = fit
        self.base = fit[0] * r ** 2 + fit[1] * r + fit[2]
        if detected:
            self.acc_failure = 0
        else:
            self.acc_failure += 1
        self.detected = detected
        # Calculate the radius of curvature (in meters) for
        # each of left & right lane lines, respectively.
        mx = conversion_params['meters_per_pixel_x']
        my = conversion_params['meters_per_pixel_y']
        sf0 = fit[0] * mx / my ** 2
        sf1 = fit[1] * mx / my
        # Calculate curvature at the bottom of the image.
        self.curverad = (1 + (2 * sf0 * r + sf1) ** 2) ** (3 / 2) \
                      / np.abs(2 * sf0)
        # Keep the last N sets of coefficients for lane line fitting.
        if self.prev_fits is not None:
            self.prev_fits = np.append(self.prev_fits, [self.curr_fit], axis=0)
        else:
            self.prev_fits = np.float32([self.curr_fit], dtype=np.float32)
        if len(self.prev_fits) > params['number_of_fit_records']:
            self.prev_fits = self.prev_fits[1:, :]

    # Calculate average fit of last N samples.
    # Expect an np.array (of y-axis coordinates) and
    # return an array (of corresponding x-axis coordinates).
    def average_fit(self, y):
        if len(self.prev_fits) > 0:
            weight_param = params['fit_avg_weights']
            weights = weight_param[::-1][:len(self.prev_fits)]
            c = np.average(self.prev_fits, axis=0, weights=weights)
        else:
            c = np.float32([0, 0, 0])
        # self.log.debug("Average coefficients: %s" % c)
        # The following calculation is on a vector.
        x = c[0] * y ** 2 + c[1] * y + c[2]

        return x

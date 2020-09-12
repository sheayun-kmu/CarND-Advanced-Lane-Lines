class LaneLine:

    def __init__(self):
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

    # Update lane line status using a polynomial fit for the
    # newly detected lane line.
    def update(self, img_size, x, y, fit, curverad, detected):
        r, c = img_size
        self.x, self.y = x, y
        self.curr_fit = fit
        self.base = fit[0] * r ** 2 + fit[1] * r + fit[2]
        if detected:
            self.acc_failure = 0
        else:
            self.acc_failure += 1
        self.detected = detected
        self.curverad = curverad

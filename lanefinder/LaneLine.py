class LaneLine:

    def __init__(self):
        # whether this line has been detected previously
        self.detected = False
        # base position (bottom) for the currently detected lane line
        self.base = 0
        # polynomial coefficients for the most recent fit
        self.curr_fit = (0, 0, 0)

    # Update lane line status using a polynomial fit for the
    # newly detected lane line.
    def update(self, img_size, fit, detected):
        r, c = img_size
        self.curr_fit = fit
        self.base = fit[0] * r ** 2 + fit[1] * r + fit[2]
        self.detected = detected

import logging
import numpy as np
import cv2

from lanefinder.params import camera_params
from lanefinder.params import perspective_params
from lanefinder.params import detector_params
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer
from lanefinder.LaneDetector import LaneDetector
from lanefinder.LaneLine import LaneLine

# Image processing pipeline
# 1. initialize
#    - camera calibration
#    - perspective transformation parameters
# 2. pipelining
#    - input: image taken by the camera (possibly from video)
#    - output: undistorted, binarized, warped image

class ImgPipeline:

    # Calibrate camera & setup perspective transformation parameters.
    def __init__(self, calib_image_files=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        # Get a calibrated camera model for distortion correction.
        self.cam = CamModel()
        if not calib_image_files:
            calib_image_files = camera_params['filepaths']
        # If we still don't have no calibration image file, it's an error.
        if not calib_image_files:
            self.log.error("No calibration image found")
            import sys
            sys.exit(-1)
        self.cam.calibrate(calib_image_files, 9, 6)
        # Initialize the camera's perspective transform.
        self.cam.init_perspective()
        # Initialize a binarizer (with default thresholds).
        self.bin = Binarizer()
        # We want to keep the undistorted version
        # (needed for later rendering of debugging display).
        self.undistorted = None
        # LaneDetector instance (used for both left & right lane lines)
        self.detector = LaneDetector()
        # Left and Right lane lines
        self.left, self.right = LaneLine(), LaneLine()
        # The following image is needed for lane detection debugging.
        self.debug_img = None

    # Getter method for the undistorted version kept in the pipeline.
    def get_undistorted(self):
        return self.undistorted

    # Undistort, binarize, and warp (to bird's eye view) image.
    def preprocess(self, img):
        undistorted = self.cam.undistort(img)
        # We keep a copy of this undistorted image.
        self.undistorted = np.copy(undistorted)
        binarized = self.bin.binarize(undistorted)
        result = self.cam.warp(binarized)
        # We keep a copy of binarized warped image (for debugging).
        self.debug_img = np.dstack((result, result, result)) * 255
        return result

    # Detect left & right lane lines and update their states;
    # a binary warped image is expected.
    def detect_lanes(self, img):
        rows, cols = img.shape[:2]
        # Calculate horizontal range to begin with, for
        # left & right lane lines, respectively.
        base_range_l = (0, cols // 2)
        base_range_r = (cols // 2, cols)
        detected_l, detected_r = False, False
        if self.left.detected:
            lx, ly, lf = self.detector.search_around_prev(img, self.left)
            detected_l = True
        else:
            lx, ly, lf = self.detector.slide_from_peak(img, base_range_l)
            if lf is not (0, 0, 0):
                detected_l = True
            else:
                self.log.debug("Lost left line - not detected around prev")
        if self.right.detected:
            rx, ry, rf = self.detector.search_around_prev(img, self.right)
            detected_r = True
        else:
            rx, ry, rf = self.detector.slide_from_peak(img, base_range_r)
            if rf is not (0, 0, 0):
                detected_r = True
            else:
                self.log.debug("Lost right line - not detected around prev")



        self.annotate_debug_img(lx, ly, lf, rx, ry, rf)




        # Sanity check #1 - whether each lane line detected is
        # close to the previously detected one.
        # If the check fails, fall back to the previously detected lane line.
        base_l = lf[0] * rows ** 2 + lf[1] * rows + lf[2]
        base_r = rf[0] * rows ** 2 + rf[1] * rows + rf[2]
        base_drift_limit = detector_params['base_drift_limit']
        if self.left.detected \
           and np.abs(base_l - self.left.base) > base_drift_limit:
            lf = self.left.curr_fit
            detected_l = False
            self.log.debug("Lost left line - too much drift"
                           " (%s vs %s)" % (base_l, self.left.base))
        if self.right.detected \
           and np.abs(base_r - self.right.base) > base_drift_limit:
            rf = self.right.curr_fit
            detected_r = False
            self.log.debug("Lost right line - too much drift"
                           " (%s vs %s)" % (base_r, self.right.base))
        # Sanity check #2 - whether the two detected lane lines are
        # approximately parallel.
        # If the check fails, discard both the detected lane lines.
        parallel_check_limit = detector_params['parallel_check_limit']
        if np.abs(lf[0] - rf[0]) > parallel_check_limit:
            lf, rf = self.left.curr_fit, self.right.curr_fit
            detected_l, detected_r = False, False
            self.log.debug("Both lines discarded - parallel check failed")
        # Now we have the currently determined lane lines
        # (though possibly fallen back to the previous ones),
        # we update the lane line status.
        self.left.update((rows, cols), lf, detected_l)
        self.right.update((rows, cols), rf, detected_r)

    # Paint drivable areas (between left & right lane lines).
    def paint_drivable(self, paint_color=(0, 255, 0)):
        img = self.undistorted
        lc, rc = self.left.curr_fit, self.right.curr_fit
        # Initialize a blank image the same size as the given.
        overlay = np.zeros_like(img, dtype=np.uint8)
        # Cacluate the second-order polynomials for
        # left & right lane line approximation.
        y = np.linspace(0, overlay.shape[0] - 1, overlay.shape[0])
        lx = lc[0] * y ** 2 + lc[1] * y + lc[2]
        rx = rc[0] * y ** 2 + rc[1] * y + rc[2]
        # Collect points on left & right (detected) lane lines.
        pts_l = np.array([np.transpose(np.vstack([lx, y]))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack([rx, y])))])
        # Concatenate them to form an outline of (detected) drivable area.
        pts = np.hstack((pts_l, pts_r))
        # Paint the drivable area on the blank image (on warped space).
        cv2.fillPoly(overlay, np.int_([pts]), paint_color)
        # Inverse-warp the painted image to form an overlay.
        unwarped = self.cam.inverse_warp(overlay)
        # Stack the two (original & painted) images.
        result = cv2.addWeighted(img, 0.7, unwarped, 0.3, 0)
        return result

    # Componse an image for lane detection debugging.
    def annotate_debug_img(self, lx, ly, lf, rx, ry, rf):
        img = self.debug_img
        r, c = img.shape[:2]
        # Color in left and right collected pixels.
        img[ly, lx] = [255, 0, 0]
        img[ry, rx] = [0, 0, 255]

        # Draw the polynomial currently fit and newly detected.
        for y in range(r):
            fit = self.left.curr_fit
            left = np.int(fit[0] * y ** 2 + fit[1] * y + fit[2])
            img[y, left - 2:left + 2] = [255, 255, 0]
            fit = self.right.curr_fit
            right = np.int(fit[0] * y ** 2 + fit[1] * y + fit[2])
            img[y, right - 2:right + 2] = [255, 255, 0]
            left = np.int(lf[0] * y ** 2 + lf[1] * y + lf[2])
            img[y, left - 2:left + 2] = [0, 255, 0]
            right = np.int(rf[0] * y ** 2 + rf[1] * y + rf[2])
            img[y, right - 2:right + 2] = [0, 255, 0]

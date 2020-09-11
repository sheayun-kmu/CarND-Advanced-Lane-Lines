import logging
import numpy as np
import cv2

from lanefinder.params import camera_params
from lanefinder.params import perspective_params
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer

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

    # Getter method for the undistorted version kept in the pipeline.
    def get_undistorted(self):
        return self.undistorted

    # Undistort, binarize, and warp (to bird's eye view) image.
    def process(self, img):
        undistorted = self.cam.undistort(img)
        # We keep a copy of this undistorted image.
        self.undistorted = np.copy(undistorted)
        binarized = self.bin.binarize(undistorted)
        result = self.cam.warp(binarized)
        return result

    # Paint drivable areas (between left & right lane lines).
    def paint_drivable(self, orig_img, overlay, left_coeffs, right_coeffs,
                       paint_color=(0, 255, 0)):
        lc, rc = left_coeffs, right_coeffs
        # Initialize a blank image the same size as the given.
        img = np.zeros_like(orig_img, dtype=np.uint8)
        # Cacluate the second-order polynomials for
        # left & right lane line approximation.
        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
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
        result = cv2.addWeighted(orig_img, 0.7, unwarped, 0.3, 0)
        return img, result

import logging
import numpy as np
import cv2

from lanefinder.params import camera_params
from lanefinder.params import perspective_params
from lanefinder.params import detector_params
from lanefinder.params import conversion_params
from lanefinder.params import display_params

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
        self.log.setLevel(logging.WARN)
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
    def detect_lanes(self, img, debug=False):
        rows, cols = img.shape[:2]
        # Calculate horizontal range to begin with, for
        # left & right lane lines, respectively.
        base_range_l = (0, cols // 2)
        base_range_r = (cols // 2, cols)
        failure_acc_limit = detector_params['failure_acc_limit']
        left_reset, right_reset = False, False
        if self.left.very_first() == True \
           or self.left.acc_failure >= failure_acc_limit:
            lx, ly, lf = self.detector.slide_from_peak(img, base_range_l)
            refresh_l = True
            if self.left.very_first() == True:
                self.log.debug("The very first detection trial for left")
            else:
                left_reset = True
                self.log.debug("Too many failures for left lane, restart")
        else:
            lx, ly, lf = self.detector.search_around_prev(img, self.left)
            refresh_l = False
        if self.right.very_first() == True \
           or self.right.acc_failure >= failure_acc_limit:
            rx, ry, rf = self.detector.slide_from_peak(img, base_range_r)
            refresh_r = True
            if self.right.very_first() == True:
                self.log.debug("The very first detection trial for right")
            else:
                right_reset = True
                self.log.debug("Too many failures for right lane, restart")
        else:
            rx, ry, rf = self.detector.search_around_prev(img, self.right)
            refresh_r = False

        # Perform sanity checks.
        if not (self.left.very_first() or self.right.very_first()) \
           and (left_reset == False and right_reset == False):
            lx, ly, lf, det_l, rx, ry, rf, det_r = self.sanity_check(
                lx, ly, lf, refresh_l, rx, ry, rf, refresh_r
            )
        else:
            det_l, det_r = refresh_l, refresh_r

        # If debug flag is set, we compose a warped perspective image
        # that is annotated with pixels contained in the lane lines.
        if debug:
            self.annotate_debug_img(lx, ly, lf, rx, ry, rf)

        # Now we have the currently determined lane lines
        # (though possibly fallen back to the previous ones),
        # we update the lane line status.
        self.left.update((rows, cols), lx, ly, lf, det_l)
        # self.log.debug("Left lane updated with %s" % lf)
        self.right.update((rows, cols), rx, ry, rf, det_r)
        # self.log.debug("Right lane updated with %s" % rf)

    def sanity_check(self, lx, ly, lf, refresh_l, rx, ry, rf, refresh_r):
        params = detector_params['sanity_check_params']
        sanity_check_base_drift = params['sanity_check_base_drift']
        sanity_check_curvature_diff = params['sanity_check_curvature_diff']
        sanity_check_lane_width = params['sanity_check_lane_width']
        sanity_check_parallel = params['sanity_check_parallel']

        rows, cols = self.debug_img.shape[:2]
        # Set both det_l and det_r to True;
        # later they can be set to False based on sanity checks.
        det_l, det_r = True, True

        # Sanity check #1 - whether each lane line detected is
        # close to the previously detected one.
        # If the check fails, fall back to the previously detected lane line.
        if sanity_check_base_drift:
            base_l = lf[0] * rows ** 2 + lf[1] * rows + lf[2]
            base_r = rf[0] * rows ** 2 + rf[1] * rows + rf[2]
            base_drift_limit = detector_params['base_drift_limit']
            if np.abs(base_l - self.left.base) > base_drift_limit:
                lx, ly, lf = self.left.x, self.left.y, self.left.curr_fit
                det_l = False
                self.log.debug("Left line discarded - too much drift"
                            " (%s vs %s)" % (base_l, self.left.base))
            if np.abs(base_r - self.right.base) > base_drift_limit:
                rx, ry, rf = self.right.x, self.right.y, self.right.curr_fit
                det_r = False
                self.log.debug("Right line discarded - too much drift"
                            " (%s vs %s)" % (base_r, self.right.base))

        # Sanity check #2 - whether curvature of each line jumps beyone
        # a certain threshold.
        if sanity_check_curvature_diff:
            bound = detector_params['curvature_diff_limit']
            if not refresh_l and np.abs(lf[0] - self.left.curr_fit[0]) > bound:
                lx, ly, lf = self.left.x, self.left.y, self.left.curr_fit
                det_l = False
                self.log.debug(
                    "Left line discarded - change in curvature too steep"
                )
            if np.abs(rf[0] - self.right.curr_fit[0]) > bound:
                rx, ry, rf = self.right.x, self.right.y, self.right.curr_fit
                det_r = False
                self.log.debug(
                    "Left line discarded - change in curvature too steep"
                )

        # Sanity check #3 - whether the distance between the lane lines
        # are within a reasonable bound.
        if sanity_check_lane_width:
            center = cols / 2
            mx = conversion_params['meters_per_pixel_x']
            my = conversion_params['meters_per_pixel_y']
            bound_l = detector_params['lane_width_lower_bound']
            bound_u = detector_params['lane_width_upper_bound']
            if np.abs(base_l - base_r) < bound_l:
                # Discard a line closer to the center.
                if np.abs(center - base_l) < np.abs(base_r - center):
                    lx, ly, lf = self.left.x, self.left.y, self.left.curr_fit
                    det_l = False
                    self.log.debug(
                        "Left lane discarded - lane too narrow (%s)" % \
                        np.abs((base_r - base_l) * mx)
                    )
                else:
                    rx, ry, rf = self.right.x, self.right.y, self.right.curr_fit
                    det_r = False
                    self.log.debug(
                        "Right lane discarded - lane too narrow (%s)" % \
                        np.abs((base_r - base_l) * mx)
                    )
            elif np.abs(base_l - base_r) > bound_u:
                # Discard a line farther from center.
                if np.abs(center - base_l) > np.abs(base_r - center):
                    lx, ly, lf = self.left.x, self.left.y, self.left.curr_fit
                    det_l = False
                    self.log.debug(
                        "Left lane discarded - lane too wide (%.2f)" % \
                        np.abs((base_r - base_l) * mx)
                    )
                else:
                    rx, ry, rf = self.right.x, self.right.y, self.right.curr_fit
                    det_r = False
                    self.log.debug(
                        "Right lane discarded - lane too wide (%.2f)" % \
                        np.abs((base_r - base_l) * mx)
                    )

        # Sanity check #4 - whether the two detected lane lines are
        # approximately parallel.
        # If the check fails, discard one that's not fresh.
        if sanity_check_parallel:
            parallel_check_limit = detector_params['parallel_check_limit']
            if np.abs(lf[0] - rf[0]) > parallel_check_limit:
                if not refresh_l:
                    lx, ly, lf = self.left.x, self.left.y, self.left.curr_fit
                    det_l = False
                    self.log.debug(
                        "Left line discarded - parallel check failed"
                    )
                if not refresh_r:
                    rx, ry, rf = self.right.x, self.right.y, self.right.curr_fit
                    det_r = False
                    self.log.debug(
                        "Left line discarded - parallel check failed"
                    )

        # Return values are as follows:
        return lx, ly, lf, det_l, rx, ry, rf, det_r

    # Paint drivable areas (between left & right lane lines).
    def paint_drivable(self, paint_color=(0, 255, 0)):
        img = self.undistorted
        lc, rc = self.left.curr_fit, self.right.curr_fit
        # Initialize a blank image the same size as the given.
        overlay = np.zeros_like(img, dtype=np.uint8)
        # Cacluate the second-order polynomials for
        # left & right lane line approximation.
        y = np.linspace(0, overlay.shape[0] - 1, overlay.shape[0])
        lx = self.left.average_fit(y)
        rx = self.right.average_fit(y)
        # Collect points on left & right (detected) lane lines.
        pts_l = np.array([np.transpose(np.vstack([lx, y]))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack([rx, y])))])
        # Concatenate them to form an outline of (detected) drivable area.
        pts = np.hstack((pts_l, pts_r))
        # Paint the drivable area on the blank image (on warped space).
        cv2.fillPoly(overlay, np.int_([pts]), paint_color)
        # Red pixels for left lane line, blue for right.
        # This is done after the green so that pixels stand out.
        overlay[self.left.y, self.left.x] = [255, 0, 0]
        overlay[self.right.y, self.right.x] = [0, 0, 255]
        # Inverse-warp the painted image to form an overlay.
        unwarped = self.cam.inverse_warp(overlay)
        # Stack the two (original & painted) images.
        result = cv2.addWeighted(img, 0.7, unwarped, 0.3, 0)
        return result

    # Annotate the resulting image with text containing the following info:
    # - radius of curvature (in meters)
    # - vehicle distance from center of the lane
    def annotate_info(self, img):
        # Fetch display parameters (from configuration)
        font = display_params['text_font']
        bottom_left = display_params['text_position']
        font_scale = display_params['font_scale']
        font_color = display_params['font_color']
        line_type = display_params['line_type']

        # Average left & right curvature radius to display.
        curve_rad = (self.left.curverad + self.right.curverad) / 2
        # Center point of two lane lines; convert to meters.
        mx = conversion_params['meters_per_pixel_x']
        center = (self.left.base + self.right.base) / 2
        offset = img.shape[1] / 2 - center
        offset_meters = offset * mx

        info_str = 'Radius of Curvature = %5s(m)' % \
                   (np.int(curve_rad) if curve_rad < 100000 else '<inf>')
        position = bottom_left
        img = cv2.putText(
            img, info_str, position,
            font, font_scale, font_color,
            line_type
        )
        info_str = 'Vehicle is %.2fm %s of center' % (
            np.abs(offset_meters), 'left' if offset < 0 else 'right'
        )
        position = (bottom_left[0], bottom_left[1] + 40)
        img = cv2.putText(
            img, info_str, position,
            font, font_scale, font_color,
            line_type
        )
        lane_width = np.abs(self.right.base - self.left.base) * mx
        info_str = "Detected Lane Width = %.2fm" % lane_width
        position = (bottom_left[0], bottom_left[1] + 80)
        img = cv2.putText(
            img, info_str, position,
            font, font_scale, font_color,
            line_type
        )

        return img

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

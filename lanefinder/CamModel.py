import numpy as np
import cv2
import logging
logging.basicConfig(
    level=logging.WARN,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Capture camera model
# 1. calibrate using a set of images of chessboards
# 2. undistort image based on calibration
# 3. warp image to get a bird's eye view of it

class CamModel:

    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        self.mtx = None
        self.dist = None

    # Given a set of chessboard images (and # of corners),
    # calibrate the camera and derive the conversion matrix.
    def calibrate(self, image_files, nx, ny):
        # Initialize empty imgpoints and objpoints
        imgpoints = []
        objpoints = []
        # Prepare uniform object points using simple arithmetic
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        self.log.info("Beginning camera calibration with"
                      " %d images" % len(image_files))
        for fname in image_files:
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, add object points and image points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
            else:
                self.log.warn("Failed to find %d * %d corners from"
                              " image %s" % (nx, ny, fname))
        # Calibrate
        self.log.info("Gathered %d sets of corners" % len(imgpoints))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[1::-1], None, None
        )
        self.log.info("Finished calibrating camera")
        self.mtx = mtx
        self.dist = dist

    # Given an image, return its undistorted version, based on
    # calibration parameters determined in calibrate()
    def undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

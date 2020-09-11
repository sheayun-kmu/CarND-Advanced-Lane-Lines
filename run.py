import logging
logging.basicConfig(
    level=logging.ERROR,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lanefinder.params import perspective_params
from lanefinder.ImgPipeline import ImgPipeline
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer
from lanefinder.LaneDetector import LaneDetector

def get_calibrated_cam():
    cam = CamModel()
    cal_images = glob.glob('./camera_cal/calibration*.jpg')
    cam.calibrate(cal_images, 9, 6)
    cam.init_perspective()
    return cam

# Juxtapose two images given by parameters.
# Default color map is 'viridis' (RGB, used by mpimg)
# while grayscale images (two dimensional) are considered.
def visual_compare(img1, title1, img2, title2, fontsize=20):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    cmap1 = 'gray' if len(img1.shape) == 2 else 'viridis'
    cmap2 = 'gray' if len(img2.shape) == 2 else 'viridis'
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=fontsize)
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=fontsize)
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.show()

def test_undistort(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    undistorted = cam.undistort(img)
    visual_compare(img, 'Original Image', undistorted, 'Undistorted Image')

def test_warp(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    undistorted = cam.undistort(img)
    src = np.array(
        [
            perspective_params['src']['ul'], # upper left
            perspective_params['src']['ur'], # upper right
            perspective_params['src']['lr'], # lower right
            perspective_params['src']['ll'], # lower left
        ],
        np.int32
    )
    dst = np.array(
        [
            perspective_params['dst']['ul'], # upper left
            perspective_params['dst']['ur'], # upper right
            perspective_params['dst']['lr'], # lower right
            perspective_params['dst']['ll'], # lower left
        ],
        np.int32
    )
    warped = cam.warp(undistorted)
    undistorted = cv2.polylines(undistorted, [src], True, (255, 0, 0), 2)
    warped = cv2.polylines(warped, [dst], True, (255, 0, 0), 2)
    visual_compare(undistorted, 'Undistorted Image', warped, 'Warped Image')

def test_binarizer(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    '''
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    undistorted = cam.undistort(img)
    binarizer = Binarizer()
    bin_image = binarizer.binarize(undistorted)
    visual_compare(img, 'Original Image', bin_image, 'Binarized Image')

def test_pipeline(imgfile):
    pipeline = ImgPipeline()
    img = mpimg.imread(imgfile)
    result = pipeline.process(img)
    visual_compare(img, 'Original Image', result, 'Pipeline Result')

def test_detector(imgfile):
    pipeline = ImgPipeline()
    img = mpimg.imread(imgfile)
    warped_binary = pipeline.process(img)
    detector_l = LaneDetector()
    detector_r = LaneDetector()
    midpoint = np.int(warped_binary.shape[1] // 2)
    c1 = detector_l.detect(warped_binary, (0, midpoint))
    r1 = detector_l.get_overlay()
    c2 = detector_r.detect(warped_binary, (midpoint, warped_binary.shape[1]))
    r2 = detector_r.get_overlay()
    img1, img2 = pipeline.paint_drivable(
        pipeline.get_undistorted(), r1 + r2, c1, c2
    )
    visual_compare(img, 'Original Image', img2, 'Lane Detected')

def detection_pipeline(img, img_pipeline, d1, d2):
    r = pipeline.process(img)
    m = img.shape[1] // 2
    c1 = d1.detect(r, (0, m))
    o1 = d1.get_overlay()
    c2 = d2.detect(r, (m, img.shape[1]))
    o2 = d2.get_overlay()
    img1, img2 = pipeline.paint_drivable(
        pipeline.get_undistorted(), o1 + o2, c1, c2
    )
    return img2

if __name__ == '__main__':
    test_img_file = './test_images/test1.jpg'
    # test_undistort(test_img_file)
    # test_binarizer(test_img_file)
    # test_warp('./test_images/straight_lines1.jpg')
    # test_warp('./test_images/straight_lines2.jpg')
    # test_warp(test_img_file)
    # test_pipeline('./test_images/straight_lines1.jpg')
    test_detector(test_img_file)

    '''
    import os
    from moviepy.editor import VideoFileClip
    test_video_file = './project_video.mp4'
    # test_video_file = './challenge_video.mp4'
    pipeline = ImgPipeline()
    detector_l = LaneDetector()
    detector_r = LaneDetector()
    output_pathname = os.path.join(os.getcwd(), "output.mp4")
    clip = VideoFileClip(test_video_file)
    output_clip = clip.fl_image(
        lambda x:detection_pipeline(
            x, pipeline, detector_l, detector_r
        )
    )
    output_clip.write_videofile(output_pathname, audio=False)
    '''

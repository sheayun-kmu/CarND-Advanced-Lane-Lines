import logging
logging.basicConfig(
    level=logging.ERROR,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lanefinder.ImgPipeline import ImgPipeline
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer

def get_calibrated_cam():
    cam = CamModel()
    cal_images = glob.glob('./camera_cal/calibration*.jpg')
    cam.calibrate(cal_images, 9, 6)
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
    pipeline = ImgPipeline(calib_img_path='./camera_cal/calibration*.jpg')
    img = mpimg.imread(imgfile)
    result = pipeline.process(img)
    visual_compare(img, 'Original Image', result, 'Pipeline Result')

if __name__ == '__main__':
    test_img_file = './test_images/test1.jpg'
    # test_undistort(test_img_file)
    # test_binarizer(test_img_file)
    # test_pipeline(test_img_file)

    import os
    from moviepy.editor import VideoFileClip
    test_video_file = './project_video.mp4'
    pipeline = ImgPipeline()
    output_pathname = os.path.join(os.getcwd(), "output.mp4")
    clip = VideoFileClip(test_video_file)
    output_clip = clip.fl_image(pipeline.process)
    output_clip.write_videofile(output_pathname, audio=False)

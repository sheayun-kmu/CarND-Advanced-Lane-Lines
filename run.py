import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lanefinder.CamModel import CamModel

cam = CamModel()
cal_images = glob.glob('./camera_cal/calibration*.jpg')
cam.calibrate(cal_images, 9, 6)
img = mpimg.imread('./test_images/test1.jpg')
result = cam.undistort(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(result)
ax2.set_title('Undistorted Image', fontsize=20)
plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
plt.show()

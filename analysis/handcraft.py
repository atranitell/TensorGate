
import numpy as np
import skimage.feature
import skimage.filters
from PIL import Image

def array_to_image(x):
    return Image.fromarray(np.uint8(x))


def image_to_array(x):
    return np.array(x, dtype=np.float)


def LBP(img):
    return skimage.feature.local_binary_pattern(img, 8, 2)


def Canny(img):
    return skimage.feature.canny(img)

def Harr(img):
    return skimage.feature.corner_harris(img, eps=1e-1, sigma=0.001)

def HOG(img):
    return skimage.feature.hog(img)

def Gabor(img):
    filr_real, filr_img = skimage.filters.gabor(img, frequency=0.6)
    return filr_img

def Sobel(img):
    return skimage.filters.sobel(img)

def SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    return kp

def read_image(imgpath):
    img = Image.open(imgpath)
    img_grey = img.convert('L')
    return img, img_grey


img, img_grey = read_image('avec2014/0001.jpg')
img_grey_np = image_to_array(img_grey)
x = SIFT(img_grey_np)
print(x)
print(x.shape)


array_to_image(x).show()
# LBP('avec2014/0001.jpg').show()

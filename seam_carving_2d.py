from skimage import data, io, filters, transform, color
import matplotlib.pyplot as plt
import skimage as skimage
import numpy as np
import time
from c_seam_carving import _seam_carving, _get_eimg
from c_seam_carving import _seam_carving_with_removed_seams
from c_seam_carving import _seam_carving_2d, _seam_carving_2d_dp

from seam_carving_backup import e_func as e_func
from seam_carving_backup import get_eimg as py_get_eimg

from skimage import img_as_ubyte

import warnings

start_time0 = time.time()
start_time = start_time0
# def seam_carving(img, eimg, num):

img = io.imread('pic3.jpg')
eimg = filters.sobel(color.rgb2gray(img))
#eimg = _get_eimg(img)   # cython get_eimg
#eimg = py_get_eimg(img, e_func)   # python get_eimg
# img = _seam_carving(img, eimg, 238,'vertical')
# img, imgwithremoval = _seam_carving_with_removed_seams(img, eimg, 200,'vertical')

dp = _seam_carving_2d_dp(img, eimg, 50, 50)
# dp = _seam_carving_2d_dp(img, eimg, 10, 10)
print("--- %s seconds to get dp ---" % (time.time() - start_time))
start_time = time.time()

img = _seam_carving_2d(img,eimg,50,50, dp)
# img = _seam_carving_2d(img,eimg,5,5, dp)

print("--- %s seconds to remove seams ---" % (time.time() - start_time))
start_time = time.time()

# filename='remove10lines_2d.png'
# io.imsave(filename, img)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    img = img_as_ubyte(img)
io.imsave('remove10lines_2d.png', img)
print("--- %s seconds to save img ---" % (time.time() - start_time))
print("--- %s seconds overall ---" % (time.time() - start_time0))



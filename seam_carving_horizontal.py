from skimage import data, io, filters, transform, color
import matplotlib.pyplot as plt
import skimage as skimage
import numpy as np
import time
start_time = time.time()
#from c_seam_carving_backup import _seam_carving, _get_eimg
from c_seam_carving import _seam_carving, _get_eimg
from c_seam_carving import _seam_carving_with_removed_seams
from seam_carving_backup import e_func as e_func
from seam_carving_backup import get_eimg as py_get_eimg

# def seam_carving(img, eimg, num):

img = io.imread('pic3.jpg')


eimg = filters.sobel(color.rgb2gray(img))
#eimg = _get_eimg(img)   # cython get_eimg
#eimg = py_get_eimg(img, e_func)   # python get_eimg
# img = _seam_carving(img, eimg, 50)
# img, imgwithremoval = _seam_carving(img, eimg, 50)
# img = _seam_carving(img, eimg, 50,'horizontal')
img, imgwithremoval = _seam_carving_with_removed_seams(img, eimg, 50,'horizontal')


print("--- %s seconds ---" % (time.time() - start_time))


io.imsave('remove200lines_horizontal.png', img)
io.imsave('remove200lines_horizontal_withremoval.png', imgwithremoval)
# plt.figure()
# plt.imshow(img)
# plt.show()


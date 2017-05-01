from skimage import data, io, filters, transform, color
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import skimage as skimage
import numpy as np
import time
start_time = time.time()
from c_seam_carving import _seam_carving, _get_eimg
from c_seam_carving import _seam_carving_with_removed_seams

from seam_carving_backup import e_func as e_func
from seam_carving_backup import get_eimg as py_get_eimg

# def seam_carving(img, eimg, num):

img = io.imread('pic3.jpg')
# img = io.imread('arch_original.png')


eimg = filters.prewitt(color.rgb2gray(img))
# eimg = filters.roberts(color.rgb2gray(img))
# eimg = filters.sobel(color.rgb2gray(img))

#eimg = _get_eimg(img)   # cython get_eimg
#eimg = py_get_eimg(img, e_func)   # python get_eimg
# img = _seam_carving(img, eimg, 238,'vertical')
img, imgwithremoval, lines = _seam_carving_with_removed_seams(img, eimg, 238,'vertical')

print("--- %s seconds ---" % (time.time() - start_time))

io.imsave('remove238lines_prewitt.png', img)
io.imsave('remove238lines_prewitt_withlines.png', imgwithremoval)
io.imsave('remove238lines_prewitt_energy.png', eimg)
# io.imsave('remove238lines_roberts.png', img)
# io.imsave('remove238lines_roberts_withlines.png', imgwithremoval)
# io.imsave('remove238lines_roberts_energy.png', eimg)
# io.imsave('remove238lines_sobel.png', img)
# io.imsave('remove238lines_sobel_withlines.png', imgwithremoval)
# io.imsave('remove238lines_sobel_energy.png', eimg)
# io.imsave('remove238lines_get_eimg.png', img)
# io.imsave('remove238lines_get_eimg_withremoval.png', imgwithremoval)
# plt.figure()
# plt.imshow(img)
# plt.show()
# plt.figure()
# plt.imshow(imgwithremoval)
# plt.show()

#written by Chang
img = io.imread('pic3.jpg')

eimg = filters.prewitt(color.rgb2gray(img))

img, imgwithEnlarge, seams = _seam_carving_with_removed_seams(img, eimg, 238,'vertical')
img = io.imread('pic3.jpg')

def _enlargeImage(img, seams, mode):

    if(mode == 'vertical'):
		# create a new image
		new_shape = list(img.shape)
		new_shape[1] += len(seams[0])
		img_new = np.zeros(new_shape, dtype=img.dtype)

		# sort the seams by index to get splits
		seams.sort(axis=1)

		for i in range(img.shape[0]):
		    split = seams[i,:]
		    # enlarge the image row-by-row via splits
		    img_new[i,0:split[0]] = img[i,0:split[0]]
		    for j in range(split.size):
		        img_new[i,split[j-1]+j:split[j]+j] = img[i,split[j-1]:split[j]]
		        img_new[i,split[j]+j] = img[i,split[j]]
		    img_new[i,split[-1]+split.size:] = img[i,split[-1]:]
    if(mode == 'horizontal'):
        img = transform.rotate(img,-90,resize=True)

	# create a new image
        new_shape = list(img.shape)
        new_shape[1] += len(seams[0])
        img_new = np.zeros(new_shape, dtype=img.dtype)

	# sort the seams by index to get splits
        seams.sort(axis=1)

        for i in range(img.shape[0]):
		    split = seams[i,:]
		    # enlarge the image row-by-row via splits
		    img_new[i,0:split[0]] = img[i,0:split[0]]
		    for j in range(split.size):
		        img_new[i,split[j-1]+j:split[j]+j] = img[i,split[j-1]:split[j]]
		        img_new[i,split[j]+j] = img[i,split[j]]
		    img_new[i,split[-1]+split.size:] = img[i,split[-1]:]

        img_new = transform.rotate(img_new,90,resize=True)
    return img_new

scaledImg = resize(img, (364, 776), mode='reflect')
io.imsave('resize238lines.png', scaledImg)

img = _enlargeImage(img, seams, 'vertical')


io.imsave('add238lines_prewitt.png', img)
#io.imsave('add238lines_prewitt_withlines.png', imgwithEnlarge)


#content amplification
img = io.imread('pic3.jpg')
lenx = img.shape[0]
leny = img.shape[1]

scaledImg = rescale(img, 1.1, mode='reflect')
#io.imsave('rescale.png', scaledImg)
eimg = filters.prewitt(color.rgb2gray(scaledImg))
#print(scaledImg.shape)

img, imgwithCA, seams = _seam_carving_with_removed_seams(scaledImg, eimg, scaledImg.shape[1] - leny,'vertical')
#io.imsave('rescaleVertical.png', img)

eimg = filters.prewitt(color.rgb2gray(img))
img, imgwithCA, seams = _seam_carving_with_removed_seams(img, eimg, scaledImg.shape[0] - lenx,'horizontal')
io.imsave('content_amplification.png', img)

def removeSeam(img, seam):
    """
        Removes a vertical seam from an input image.
    """
    # create a new image with the seam removed
    new_shape = list(img.shape)
    new_shape[1] -= 1
    img_new = np.zeros(new_shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        img_new[i,0:seam] = img[i,0:seam]
        img_new[i,seam:] = img[i,seam+1:]

    return img_new

#removing columns with minimal energy
img = io.imread('pic3.jpg')

for n in range(238):
    eimg = filters.prewitt(color.rgb2gray(img))
    lenx = eimg.shape[0]
    leny = eimg.shape[1]
    index = np.sum(eimg, axis = 0)
    arr = np.array(index)
    cols = np.argsort(arr) #index of col need to be removed
    img_next = removeSeam(img, cols[2])
    img = img_next

io.imsave('removeCols.png', img)      

'''
img = io.imread('pic3.jpg')
cropped = img[0:364,200:500]
io.imsave('crop.png', cropped)
'''

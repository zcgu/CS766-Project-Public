from Tkinter import *
from PIL import Image, ImageTk

from scipy.misc import imread, imsave
from skimage import filters, color
import numpy as np
from seam_carving_lib import get_remove_order
from c_seam_carving import _add_seams



def cvaction(event):
    w =  event.width
    print w

    if w == imgWidth:
        cv.delete("all")
        cv.create_image(0, 0, image=originpic, anchor='nw')
        return

    if w < imgWidth:

        num_to_remove = imgWidth - w
        rest = remove_map >= num_to_remove
        rest.reshape(imgHeight,imgWidth)

        nimgr = img[:,:,0][rest]
        nimgg = img[:,:,1][rest]
        nimgb = img[:,:,2][rest]
        nimgr.resize(imgHeight,w)
        nimgg.resize(imgHeight,w)
        nimgb.resize(imgHeight,w)
        nimg = np.stack((nimgr,nimgg,nimgb),axis=-1)

        picImage = Image.fromarray(nimg)
        cv.delete("all")
        originpic.paste(picImage,box=None)
        cv.create_image(0, 0, image=originpic, anchor='nw')
        print "new image shown %d"%w
        return

    if w > imgWidth:
        num_to_add = w - imgWidth
#         shape = (imgHeight, w, 3)
#         nimg = np.zeros(shape=shape,dtype='uint8')
#         imgptr = np.zeros(shape=(imgHeight),dtype='int32')
#         nimgptr = np.zeros(shape=(imgHeight),dtype='int32')
#         for y in range(0,364):
#             for x in range(0,538):
#                 nimg[y, nimgptr[y], :] = img[y, imgptr[y], :]
#                 nimgptr[y] +=1
#                 if remove_map[y,x] < num_to_add:
#                     nimg[y, nimgptr[y], :] = img[y, imgptr[y], :]
#                     nimgptr[y]+=1
#                 imgptr[y] +=1
        # too slow, use cython

        nimg = _add_seams(img, num_to_add, remove_map)

        picImage = Image.fromarray(nimg)
        cv.delete("all")
        originpic.paste(picImage,box=None)
        cv.create_image(0, 0, image=originpic, anchor='nw')
        print "new image shown %d" % w
        return


if __name__ == "__main__":

    root = Tk()
    filename='remove.jpg'
    img = imread(filename)
    remove_map = get_remove_order(img)
    print "got map"


    imgWidth= len(img[0])
    imgHeight= len(img);

    canvas = np.zeros(shape=(imgHeight,imgWidth*2,3),dtype='uint8')
    originpicImage = Image.fromarray(canvas)
    originpic = ImageTk.PhotoImage(originpicImage)

    cv = Canvas(width=imgWidth,height=imgHeight)
    cv.pack(side='top', fill='both', expand='yes')
    cv.create_image(0, 0, image=originpic, anchor='nw')
    cv.bind("<Configure>",cvaction)

    mainloop()


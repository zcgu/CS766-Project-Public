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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# def seam_carving(img, eimg, num):

img = io.imread('remove.jpg')
# img = io.imread('arch_original.png')


###############################################################################################
# Rectangle.
###############################################################################################

# class Annotate(object):
#     def __init__(self):
#         self.ax = plt.gca()
#         self.rect = Rectangle((0,0), 1, 1)
#         self.x0 = None
#         self.y0 = None
#         self.x1 = None
#         self.y1 = None
#         self.ax.add_patch(self.rect)
#         self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

#     def on_press(self, event):
#         print 'press'
#         self.x0 = event.xdata
#         self.y0 = event.ydata

#     def on_release(self, event):
#         print 'release'
#         self.x1 = event.xdata
#         self.y1 = event.ydata
#         self.rect.set_width(self.x1 - self.x0)
#         self.rect.set_height(self.y1 - self.y0)
#         self.rect.set_xy((self.x0, self.y0))
#         self.ax.figure.canvas.draw()


# plt.figure()
# a = Annotate()
# plt.imshow(img)
# plt.show()

# if a.x1 is not None:

#     eimg = filters.prewitt(color.rgb2gray(img))
#     # eimg = filters.roberts(color.rgb2gray(img))
#     # eimg = filters.sobel(color.rgb2gray(img))

#     x0, x1, y0, y1 = int(a.x0), int(a.x1), int(a.y0), int(a.y1)
#     print np.shape(eimg), x0, x1, y0, y1
#     eimg[y0: y1, x0: x1] = - (2 ** 29)

#     #eimg = _get_eimg(img)   # cython get_eimg
#     #eimg = py_get_eimg(img, e_func)   # python get_eimg
#     # img = _seam_carving(img, eimg, 238,'vertical')
#     img, imgwithremoval, lines = _seam_carving_with_removed_seams(img, eimg, x1 - x0,'vertical')

#     print("--- %s seconds ---" % (time.time() - start_time))

#     # io.imsave('remove238lines_prewitt.png', img)
#     # io.imsave('remove238lines_prewitt_withlines.png', imgwithremoval)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
#     plt.figure()
#     plt.imshow(imgwithremoval)
#     plt.show()
###############################################################################################

from pylab import *
from matplotlib.path import Path
import matplotlib.patches as patches


class ROI:

    def __init__(self, fig, data):
        self.previous_point = []
        self.start_point = []
        self.end_point = []
        self.line = None    

        self.fig =  fig
        self.fig.canvas.draw()

        self.verts = []
        self.data = data

    def button_press_callback(self, event):
        if event.inaxes: 
            x, y = event.xdata, event.ydata
            axes = event.inaxes

            self.verts.append((x, y))

            # if (event.key == 'shift') and (event.button == 1):  # If you press the right button
            if event.button == 1:  # If you press the right button
                if self.line == None: # if there is no line, create a line
                    self.line = Line2D([x,  x],
                                       [y, y],
                                       marker = 's')
                    self.start_point = [x,y]
                    self.previous_point =  self.start_point
                    axes.add_line(self.line)
                    self.fig.canvas.draw()
                # add a segment
                else: # if there is a line, create a segment
                    self.line = Line2D([self.previous_point[0], x],
                                       [self.previous_point[1], y],
                                       marker = 'o')
                    self.previous_point = [x,y]
                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()



            if (event.button == 3) and (self.line != None): # close the loop
                self.line.set_data([self.previous_point[0], self.start_point[0]],
                                   [self.previous_point[1], self.start_point[1]])                       
                axes.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None

                path1 = Path(self.verts)
                # index = path1.contains_points(self.data)

                sizex = np.shape(self.data)[0]
                sizey = np.shape(self.data)[1]
                lst = np.array([[x + 1, y + 1] for x in range(sizex) for y in range(sizey)])
                index = path1.contains_points(lst)
                # print lst[index]

                img = self.data
                eimg = filters.prewitt(color.rgb2gray(img))
                # eimg = filters.roberts(color.rgb2gray(img))
                # eimg = filters.sobel(color.rgb2gray(img))

                for y, x in lst[index]:
                    eimg[x, y] = - (2 ** 16)

                #eimg = _get_eimg(img)   # cython get_eimg
                #eimg = py_get_eimg(img, e_func)   # python get_eimg
                # img = _seam_carving(img, eimg, 238,'vertical')
                img, imgwithremoval, lines = _seam_carving_with_removed_seams(img, eimg, 1000,'vertical', is_object_remove=True)

                savefig('object_remove_1.png', bbox_inches='tight')
                io.imsave('object_remove_2.png', img)
                io.imsave('object_remove_3.png', imgwithremoval)
                plt.figure()
                plt.imshow(img)
                plt.show()
                plt.figure()
                plt.imshow(imgwithremoval)
                plt.show()

def main():
    plt.close("all")
    fig = plt.figure()
    plt.imshow(img)
    cursor = ROI(fig, img)

    fig.canvas.mpl_connect('button_press_event', cursor.button_press_callback)
    show()

if __name__ == "__main__":
    main()   



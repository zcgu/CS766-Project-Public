
def get_remove_order(img):
    from skimage import io, filters, color
    from c_seam_carving import _seam_carving_multi_size_map

#     img = io.imread('pic3.jpg')
    eimg = filters.prewitt(color.rgb2gray(img))
    return _seam_carving_multi_size_map(img, eimg)

if __name__ == "__main__":
    from scipy.misc import imread, imsave
    filename='remove.jpg'
    img = imread(filename)
    remove_map = get_remove_order(img)
    print remove_map


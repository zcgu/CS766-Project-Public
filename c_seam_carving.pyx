# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cdef cnp.double_t DBL_MAX = np.finfo(np.double).max

cdef void _cal_total_energy(cnp.double_t[:, ::1] eimg,
                            cnp.double_t[:, ::1] dp_e,
                            Py_ssize_t[:, ::1] dp_track,
                            Py_ssize_t cols) nogil:

    cdef Py_ssize_t x, y, j2
    cdef Py_ssize_t lenx = eimg.shape[0]
    cdef Py_ssize_t leny = eimg.shape[1]
    cdef cnp.double_t min_cost = DBL_MAX

    # for y in range(leny):
    #     dp_e[0, y] = eimg[0, y]

    for x in range(1, lenx):
        for y in range(cols):
            min_cost = DBL_MAX

            for j2 in range(y - 1, y + 2):
                if j2 >= cols or j2 < 0: continue

                if dp_e[x - 1, j2] < min_cost:
                    min_cost = dp_e[x - 1, j2]
                    dp_track[x, y] = j2

            dp_e[x, y] = min_cost + eimg[x, y]


cdef void _cal_line(Py_ssize_t start_index,
                    Py_ssize_t[::1] line,
                    Py_ssize_t[:, ::1] dp_track) nogil:
    cdef Py_ssize_t x
    cdef Py_ssize_t y = start_index
    for x in range(dp_track.shape[0] - 1, -1, -1):
        line[x] = y
        y = dp_track[x, y]


cdef void _remove_line(cnp.double_t[:, :, ::1] img,
                       cnp.double_t[:, ::1] eimg,
                       cnp.double_t[:, ::1] dp_e,
                       Py_ssize_t[:, ::1] dp_track,
                       Py_ssize_t[::1] line,
                       Py_ssize_t cols) nogil:
    lenx = eimg.shape[0]
    cdef Py_ssize_t x, y

    for x in range(lenx):
        for y in range(line[x], cols - 1):
            img[x, y] = img[x, y + 1]
            eimg[x, y] = eimg[x, y + 1]


cdef void _remove_line_with_removal(cnp.double_t[:, :, ::1] img,
                       cnp.double_t[:, ::1] location_img,
                       cnp.double_t[:, ::1] eimg,
                       cnp.double_t[:, ::1] dp_e,
                       Py_ssize_t[:, ::1] dp_track,
                       Py_ssize_t[::1] line,
                       Py_ssize_t cols) nogil:
    lenx = eimg.shape[0]
    cdef Py_ssize_t x, y

    for x in range(lenx):
        for y in range(line[x], cols - 1):
            img[x, y] = img[x, y + 1]
            eimg[x, y] = eimg[x, y + 1]
            location_img[x, y] = location_img[x, y + 1]


def _seam_carving(img, eimg, num, mode):

    from skimage import img_as_float
    img = img_as_float(img).copy()
    if ( mode == 'horizontal'):
        from skimage import transform
        img = transform.rotate(img,-90,resize=True)
        eimg = transform.rotate(eimg,-90,resize=True)

    eimg = eimg.copy()
    lenx, leny = eimg.shape
    cols = leny

    cdef cnp.double_t[:, ::1] dp_e = np.zeros(eimg.shape[0:2], dtype=np.float)
    cdef Py_ssize_t[:, ::1] dp_track = np.zeros(eimg.shape[0:2], dtype=np.intp)
    cdef Py_ssize_t[::1] line = np.zeros((eimg.shape[0]), dtype=np.intp)

    for cols in range(leny, leny - num, -1):
        _cal_total_energy(eimg, dp_e, dp_track, cols)
        y = np.argmin(dp_e[lenx - 1, :cols])
        _cal_line(y, line, dp_track)
        _remove_line(img, eimg, dp_e, dp_track, line, cols)

    img = img[:, :cols]
    if ( mode == 'horizontal'):
        img = transform.rotate(img[:,:cols],90,resize=True)
    #return img[:, :cols], img_with_removal
    return img

def _seam_carving_with_removed_seams(img, eimg, num, mode, is_object_remove=False):
    
    from skimage import img_as_float
    from skimage import transform
    img = img_as_float(img).copy()
#     img = transform.rotate(img,0,resize=True)
    # img = img*1.0;
    # img = np.asarray(img)
    if ( mode == 'horizontal'):
        img = transform.rotate(img,-90,resize=True)
        eimg = transform.rotate(eimg,-90,resize=True)

    eimg = eimg.copy()
    leny, lenx = eimg.shape
    cols = lenx

    # contruct an image-sized the matrix to record the line that is removed.
    img_with_removal = img.copy()
    location_img = np.zeros(eimg.shape[0:2], dtype=np.float);
    location_img[:,0] = 0
    for x in range(1,lenx):
        location_img[:,x] = location_img[:,x-1]+1
    removed_lines = np.zeros([leny,num],dtype=np.intp)

    cdef cnp.double_t[:, ::1] dp_e = np.zeros(eimg.shape[0:2], dtype=np.float)
    cdef Py_ssize_t[:, ::1] dp_track = np.zeros(eimg.shape[0:2], dtype=np.intp)
    cdef Py_ssize_t[::1] line = np.zeros((eimg.shape[0]), dtype=np.intp)

    for cols in range(lenx, lenx - num, -1):
        _cal_total_energy(eimg, dp_e, dp_track, cols)
        x = np.argmin(dp_e[leny - 1, :cols])

        if (is_object_remove and dp_e[leny - 1, x] > - (2 ** 10)):
            break

        _cal_line(x, line, dp_track)
        
        # record the pixel removed
        # print eimg.shape[0]
        for iy in range(0, leny):
            removed_lines[iy,lenx-cols] = location_img[iy,line[iy]]
        
        _remove_line_with_removal(img, location_img, eimg, dp_e, dp_track, line, cols)

    for y in range(0, leny):
        for n in range(0,num):
            img_with_removal[y,removed_lines[y,n]] = [1,0,0]
    
    img = img[:,:cols]
    if ( mode == 'horizontal'):
        img = transform.rotate(img,90,resize=True)
        img_with_removal = transform.rotate(img_with_removal,90,resize=True)

    return img, img_with_removal, removed_lines

def _seam_carving_multi_size_map(img, eimg):
    
    from skimage import img_as_float
    from skimage import transform
    img = img_as_float(img).copy()

    eimg = eimg.copy()
    leny, lenx = eimg.shape
    cols = lenx

    # contruct an image-sized the matrix to record the line that is removed.
    img_remove_map = np.zeros(eimg.shape[0:2], dtype=np.intp);
    location_img = np.zeros(eimg.shape[0:2], dtype=np.float);
    location_img[:,0] = 0
    for x in range(1,lenx):
        location_img[:,x] = location_img[:,x-1]+1
    removed_lines = np.zeros([leny,lenx],dtype=np.intp)

    cdef cnp.double_t[:, ::1] dp_e = np.zeros(eimg.shape[0:2], dtype=np.float)
    cdef Py_ssize_t[:, ::1] dp_track = np.zeros(eimg.shape[0:2], dtype=np.intp)
    cdef Py_ssize_t[::1] line = np.zeros((eimg.shape[0]), dtype=np.intp)

    for cols in range(lenx, 0, -1):
        _cal_total_energy(eimg, dp_e, dp_track, cols)
        x = np.argmin(dp_e[leny - 1, :cols])

        _cal_line(x, line, dp_track)
        
        # record the pixel removed
        # print eimg.shape[0]
        for iy in range(0, leny):
            removed_lines[iy,lenx-cols] = location_img[iy,line[iy]]
        
        _remove_line_with_removal(img, location_img, eimg, dp_e, dp_track, line, cols)

    for y in range(0, leny):
        for n in range(0,lenx):
            img_remove_map[y,removed_lines[y,n]] = n
    
    img = img[:,:cols]

    return img_remove_map

def _seam_carving_2d(img, eimg, row_num, col_num, dp):
    from skimage import transform
    from skimage import img_as_float
    img = img_as_float(img).copy()
    eimg = eimg.copy()
#     print "in "
#     print("size of img %d %d %d"%(img.shape))
#     print("size of eimg %d %d "%(eimg.shape))

    # contruct the removal order from dp
    row = max(0, row_num-1)
    col = max(0,col_num-1)
    order=[]
    while row != 0 or col != 0:
#         print("row %d, col %d"%(row, col))
        choice = dp[row,col]
        order.append(choice)
        row = row - (choice==1)
        col = col - (choice==0)

    cdef cnp.double_t[:, ::1] dp_e = np.zeros(eimg.shape[0:2], dtype=np.float)
    cdef Py_ssize_t[:, ::1] dp_track = np.zeros(eimg.shape[0:2], dtype=np.intp)
    cdef Py_ssize_t[::1] line = np.zeros((eimg.shape[0]), dtype=np.intp)
    cdef cnp.double_t[:, ::1] dp_e_rot = np.zeros([eimg.shape[1],eimg.shape[0]], dtype=np.float)
    cdef Py_ssize_t[:, ::1] dp_track_rot = np.zeros([eimg.shape[1],eimg.shape[0]], dtype=np.intp)
    cdef Py_ssize_t[::1] line_rot = np.zeros((eimg.shape[1]), dtype=np.intp)

    for choice in order:
#         line = np.zeros((eimg.shape[0]), dtype=np.intp)
#         print choice
#         print "--------"
#         print img.shape
#         print eimg.shape
#         print "--------"
        if choice == 1:
#             print("remove a row line")
            # rotate image
#             img_rot = transform.rotate(img,-90,resize=True).copy()
#             eimg_rot = transform.rotate(eimg,-90,resize=True).copy()
            img_rot = np.rot90(img,-1).copy(order='C')
            eimg_rot = np.rot90(eimg,-1).copy(order='C')
#             print img.shape
#             print img_rot.shape
#             print eimg.shape
#             print eimg_rot.shape
          
            #remove line
            leny_rot, lenx_rot = eimg_rot.shape
#             print "cal total energy"
            _cal_total_energy(eimg_rot, dp_e_rot, dp_track_rot, lenx_rot)
#             print "get min x"
            x_rot = np.argmin(dp_e_rot[leny_rot - 1, :lenx_rot])
#             print "cal line"
            _cal_line(x_rot, line_rot, dp_track_rot)
#             print "remove line"
            _remove_line(img_rot, eimg_rot, dp_e_rot, dp_track_rot, line_rot, lenx_rot)
            img_rot = img_rot[:,:lenx_rot-1,:].copy(order='C')
            eimg_rot = eimg_rot[:,:lenx_rot-1].copy(order='C')
            
            # rotate back
#             img = transform.rotate(img_rot,90,resize=True).copy()
#             eimg = transform.rotate(eimg_rot,90,resize=True).copy()
            img = np.rot90(img_rot,1).copy(order='C')
            eimg = np.rot90(eimg_rot,1).copy(order='C')
        else:
#             print("remove a column line")
#             print img.shape

            #remove line
            leny, lenx = eimg.shape
#             print "cal total energy"
            _cal_total_energy(eimg, dp_e, dp_track, lenx)
#             print "get min x"
            x = np.argmin(dp_e[leny - 1, :lenx])
#             print "cal line"
            _cal_line(x, line, dp_track)
#             print "remove line"
            _remove_line(img, eimg, dp_e, dp_track, line, lenx)
            img = img[:,:lenx-1,:].copy(order='C')
            eimg = eimg[:,:lenx-1].copy(order='C')

    return img


def _seam_carving_2d_dp(img, eimg, row_num, col_num):
    '''
        Return a dp table that stores energy of current img
        construct dp_remove from the left-top to right-bottom
        _seam_carving_2d(img, eimg, row_num, col_num)
    '''
    from skimage import transform
    # construct dp_remove_along_row
    img = transform.rotate(img,0,resize=True)
    eimg = transform.rotate(eimg,0,resize=True)
#     print 'calculating dp row'
    dp_remove_along_row = get_order_dp(img, eimg, row_num, col_num)
#     print 'dp row done'

#     print dp_remove_along_row
    # construct dp_remove_along_col
    img = transform.rotate(img,90,resize=True)
    eimg = transform.rotate(eimg,90,resize=True)
    img = np.flip(img,2)
    eimg = np.flip(eimg,1)
#     print 'calculating dp col'
    dp_remove_along_col = get_order_dp(img, eimg, col_num, row_num)
#     print 'dp col done'
    dp_remove_along_col = np.flip(dp_remove_along_col,1)
    dp_remove_along_col = transform.rotate(dp_remove_along_col,-90,resize=True)

    dp_remove = np.zeros([row_num, col_num],dtype=np.intp)
    # 0 from left, 1 from top
    dp_remove[0,:]=0
    dp_remove[:,0]=1
    dp_remove[0,0]=-1
    for col in range(1,col_num):
        for row in range(1,row_num):
            if dp_remove_along_col[col,row-1] > dp_remove_along_row[col-1,row]:
                dp_remove[col, row] = 0
            else:
                dp_remove[col, row] = 1
#     print 'combine dp done'

    return dp_remove


def get_order_dp(img, eimg, row_num, col_num):
    from skimage import transform
    cdef cnp.double_t[:, ::1] dp_e
    cdef Py_ssize_t[:, ::1] dp_track
    cdef Py_ssize_t[::1] line
    dp_remove_along_row = np.zeros([row_num, col_num],dtype=np.float)
    # construct dp_remove_along_row
    col_img_0 = img.copy()
    col_eimg_0 = eimg.copy()
    for row in range(0, row_num):
#         print(" %d row "%row)
        # do col removal first
        col_img = col_img_0.copy()
        col_eimg = col_eimg_0.copy()
        dp_e = np.zeros(col_eimg.shape[0:2], dtype=np.float)
        dp_track = np.zeros(col_eimg.shape[0:2], dtype=np.intp)
        line = np.zeros((col_eimg.shape[0]), dtype=np.intp)
        leny, lenx = col_eimg.shape
        for cols in range(lenx, lenx - col_num, -1):
            _cal_total_energy(col_eimg, dp_e, dp_track, cols)
            x = np.argmin(dp_e[leny - 1, :cols])
            dp_remove_along_row[row, lenx - cols] = np.min(dp_e[leny-1,:cols])
            _cal_line(x, line, dp_track)
            _remove_line(col_img, col_eimg, dp_e, dp_track, line, cols)
#         print("Done with this row, %s cols removed "%(lenx-cols))
        # do 1 seam row removal next
        col_img = transform.rotate(col_img_0,-90,resize=True)
        col_eimg = transform.rotate(col_eimg_0,-90,resize=True)
        dp_e = np.zeros(col_eimg.shape[0:2], dtype=np.float)
        dp_track = np.zeros(col_eimg.shape[0:2], dtype=np.intp)
        line = np.zeros((col_eimg.shape[0]), dtype=np.intp)
        leny, lenx = col_eimg.shape
#         print("cal total energy")
        _cal_total_energy(col_eimg, dp_e, dp_track, lenx)
#         print("get min x")
        x = np.argmin(dp_e[leny - 1, :lenx])
#         print("save to dp")
        dp_remove_along_row[0, row] = np.min(dp_e[leny-1,:lenx])
#         print("cal line")
        _cal_line(x, line, dp_track)
#         print("remove line")
        _remove_line(col_img, col_eimg, dp_e, dp_track, line, lenx)
        col_img_0 = transform.rotate(col_img,90,resize=True)[:leny-1,:]
        col_eimg_0 = transform.rotate(col_eimg,90,resize=True)[:leny-1,:]
#         print("The %s row removed "%(row))
        # do 1 seam row removal next
    return dp_remove_along_row



cdef cnp.double_t e_func(cnp.double_t[:, :, ::1] img,
                        Py_ssize_t x,
                        Py_ssize_t y ) nogil:
    '''
    Calculate the energy function
    '''

    cdef cnp.double_t res = 0.0;

    if x > 0:
        for i in range(3):
            res += dabs(float(img[x, y, i]) - float(img[x - 1, y, i]))

    if y < img.shape[1] - 1:
        for i in range(3):
            res += dabs(float(img[x, y, i]) - float(img[x, y + 1, i]))

    return res

cdef cnp.double_t dabs(cnp.double_t x) nogil:
    ''' nogil abs '''
    if x < 0: return -x
    else: return x


cdef cnp.double_t[:,::1] get_eimg(cnp.double_t[:,:,::1] img):
    cdef Py_ssize_t lenx = img.shape[0]
    cdef Py_ssize_t leny = img.shape[1]

    cdef cnp.double_t[:, ::1] eimg = np.zeros((lenx, leny), dtype=np.float)
    for x in range(1, lenx):
        for y in range(leny):
            eimg[x][y] = e_func(img, x, y)
    return eimg

def _get_eimg(img):
    return get_eimg(img)


cdef cnp.ndarray[cnp.uint8_t,ndim=3] add_seams(cnp.uint8_t[:,:,::1] img,
                                            Py_ssize_t num,
                                            Py_ssize_t[:,::1] rmmap,
                                            cnp.ndarray[cnp.uint8_t,ndim=3] nimg,
                                            Py_ssize_t[::1] imgptr,
                                            Py_ssize_t[::1] nimgptr) :
    cdef Py_ssize_t x, y

    cdef Py_ssize_t height, width
    height = img.shape[0]
    width = img.shape[1]

    for y in range(0,height):
        for x in range(0,width):
            nimg[y, nimgptr[y], :] = img[y, imgptr[y], :]
            nimgptr[y] +=1
            if rmmap[y,x] < num:
                nimg[y, nimgptr[y], :] = img[y, imgptr[y], :]
                nimgptr[y]+=1
            imgptr[y] +=1
    return nimg


def _add_seams(img,num,rmmap):
    '''
    img: original img
    num: num of seams to duplicate
    rmmap: remove order map
    '''
    
    height,width = img.shape[0:2]
    shape = (height, width+num, 3)

#     cdef cnp.uint8_t[:,:,::1] nimg = np.zeros(shape=shape,dtype='uint8')
    cdef cnp.ndarray[cnp.uint8_t,ndim=3] nimg = np.zeros(shape=shape,dtype=np.uint8)
    cdef Py_ssize_t[::1]  imgptr = np.zeros((img.shape[0]), dtype=np.intp)
    cdef Py_ssize_t[::1] nimgptr = np.zeros((img.shape[0]), dtype=np.intp)
    
    print "shape after def "
    print img.shape
    print len(nimg)

    return add_seams(img,num,rmmap,nimg,imgptr,nimgptr)
#     print "after add "
#     print nimg.shape
#     
#     return nimg



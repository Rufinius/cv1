import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import griddata
from utils import *
from matplotlib import pyplot as plt

######################
# Basic Lucas-Kanade #
######################

def compute_derivatives(im1, im2):
    """Compute dx, dy and dt derivatives.
    
    Args:
        im1: first image
        im2: second image
    
    Returns:
        Ix, Iy, It: derivatives of im1 w.r.t. x, y and t
    """
    assert im1.shape == im2.shape

    # create derivative filters (central differences)
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()

    # calculate x and y derivatives
    Ix = convolve(im1, fx, mode='mirror')
    Iy = convolve(im1, fy, mode='mirror')

    # calculate t derivative by forward difference (lecture 4 slide 41)
    It = im2 - im1
    
    assert Ix.shape == im1.shape and \
           Iy.shape == im1.shape and \
           It.shape == im1.shape

    return Ix, Iy, It


def compute_motion(Ix, Iy, It, patch_size=15, aggregate="const", sigma=2):
    """Computes one iteration of optical flow estimation.
    
    Args:
        Ix, Iy, It: image derivatives w.r.t. x, y and t
        patch_size: specifies the side of the square region R in Eq. (1)
        aggregate: 0 or 1 specifying the region aggregation region
        sigma: if aggregate=='gaussian', use this sigma for the Gaussian kernel
    Returns:
        u: optical flow in x direction
        v: optical flow in y direction
    
    All outputs have the same dimensionality as the input
    """
    assert Ix.shape == Iy.shape and \
            Iy.shape == It.shape

    weights = np.ones((patch_size, patch_size))
    # modification for task 8
    if aggregate == 'gaussian':
        weights = gaussian_kernel(patch_size, sigma)

    # obtain elements of matrices using convolution
    m1_11 = convolve(Ix*Ix, weights, mode="mirror")
    m1_12 = convolve(Ix*Iy, weights, mode="mirror")
    m1_22 = convolve(Iy*Iy, weights, mode="mirror")
    m2_1 = convolve(It*Ix, weights, mode="mirror")
    m2_2 = convolve(It*Iy, weights, mode="mirror")

    # solve system of linear equations (using manual computed inverse of 2d matrix)
    det = m1_11 * m1_22 - m1_12 * m1_12
    u = (-m1_22 * m2_1 + m1_12 * m2_2) / det
    v = (m1_12 * m2_1 - m1_11 * m2_2) / det

    """
    old computation without numpy (modified because too slow and not so nice to integrate gaussian weights
    
    # pad Ix, Iy, It using mirror (reflect) mode to allow full patch_size for border pixels
    pad = int(patch_size / 2)
    Ix_pad = np.pad(array=Ix, pad_width=pad, mode='reflect')
    Iy_pad = np.pad(array=Iy, pad_width=pad, mode='reflect')
    It_pad = np.pad(array=It, pad_width=pad, mode='reflect')

    for x in range(pad, Ix.shape[0]+pad):
        for y in range(pad, Ix.shape[1]+pad):
            # compute u and v for each pixel coordinate (x, y)

            # first compute structure tensor (m1) and matrix (m2) (-> It * [Ix, Iy])
            # by summing over all values in the patch
            m1 = np.zeros((2, 2))
            m2 = np.zeros((2, 1))
            for xp in range(x-pad, x+pad+1):
                for yp in range(y-pad, y+pad+1):
                    Ix_elem = Ix_pad[xp, yp]
                    Iy_elem = Iy_pad[xp, yp]
                    It_elem = It_pad[xp, yp]
                    m1_elem = np.array([[Ix_elem**2, Ix_elem*Iy_elem], [Ix_elem*Iy_elem, Iy_elem**2]])
                    m2_elem = np.array([[It_elem * Ix_elem], [It_elem*Iy_elem]])

                    m1 = np.add(m1, weights[xp-x+pad, yp-y+pad] * m1_elem)
                    m2 = np.add(m2, weights[xp-x+pad, yp-y+pad] * m2_elem)

            # now compute u and v by formula in lecture 10 slide 69
            uv = -1 * np.linalg.inv(m1) @ m2

            # assign values to u and v vector
            u[x-pad, y-pad] = uv[0]
            v[x-pad, y-pad] = uv[1]
    """

    assert u.shape == Ix.shape and \
            v.shape == Ix.shape
    return u, v

def warp(im, u, v):
    """Warping of a given image using provided optical flow.
    
    Args:
        im: input image
        u, v: optical flow in x and y direction
    
    Returns:
        im_warp: warped image (of the same size as input image)
    """
    assert im.shape == u.shape and \
            u.shape == v.shape

    # first transform point locations using the optical flow u, v
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    x_transformed = x + u
    y_transformed = y + v
    points = np.stack([x_transformed.flatten(), y_transformed.flatten()], axis=1)

    """
    old computation modified because too slow without numpy
    points = None
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            if points is None:
                points = np.array([[x+u[y, x], y+v[y, x]]])
            else:
                points = np.append(points, np.array([[x+u[y, x], y+v[y, x]]]), axis=0)
    """
    # now obtain the warped image using the transformed points and their corresponding values from the image
    # interpolate the missing values
    im_warp = griddata(points, im.flatten(), (x, y), method="linear", fill_value=0)

    assert im_warp.shape == im.shape
    return im_warp

def compute_cost(im1, im2):
    """Implementation of the cost minimised by Lucas-Kanade."""
    assert im1.shape == im2.shape

    d = np.linalg.norm(np.subtract(im2, im1).flatten())**2

    assert isinstance(d, float)
    return d

####################
# Gaussian Pyramid #
####################

#
# this function implementation is intentionally provided
#
def gaussian_kernel(fsize, sigma):
    """
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: deviation of the Guassian

    Returns:
        kernel: (fsize, fsize) Gaussian (normalised) kernel
    """

    _x = _y = (fsize - 1) / 2
    x, y = np.mgrid[-_x:_x + 1, -_y:_y + 1]
    G = np.exp(-0.5 * (x**2 + y**2) / sigma**2)

    return G / G.sum()

def downsample_x2(x, fsize=5, sigma=1.4):
    """
    Downsampling an image by a factor of 2
    Hint: Don't forget to smooth the image beforhand (in this function).

    Args:
        x: image as numpy array (H x W)
        fsize and sigma: parameters for Guassian smoothing
                         to apply before the subsampling
    Returns:
        downsampled image as numpy array (H/2 x W/2)
    """
    smoothed = convolve(x, gaussian_kernel(fsize, sigma), mode="mirror")

    # procedure taken from assignment 2 problem 1:
    factor = 2
    # initialize new downsampled image matrix
    H_new = int(smoothed.shape[0] / factor)
    W_new = int(smoothed.shape[1] / factor)
    downsample = np.zeros((H_new, W_new))
    for i in range(H_new):
        for j in range(W_new):
            # only take every second pixel value and ignore others
            downsample[i, j] = smoothed[i * factor, j * factor]

    return downsample


def gaussian_pyramid(img, nlevels=3, fsize=5, sigma=1.4):
    '''
    A Gaussian pyramid is a sequence of downscaled images
    (here, by a factor of 2 w.r.t. the previous image in the pyramid)

    Args:
        img: face image as numpy array (H * W)
        nlevels: num of level Gaussian pyramid, in this assignment we will use 3 levels
        fsize: gaussian kernel size, in this assignment we will define 5
        sigma: sigma of guassian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of gaussian downsampled images in ascending order of resolution
    '''

    # procedure taken from assignment 2 problem 1:
    GP = []
    # append the original image as the first level of the gaussian pyramid
    GP.append(img)
    # add the next levels of the pyramid
    for n in range(nlevels - 1):
        # smooth and downsample to obtain the next level of the gaussian pyramid
        downsampled = downsample_x2(GP[-1])
        GP.append(downsampled)

    # change order of images in pyramid
    GP_out = []
    for i in range(len(GP)):
        GP_out.append(GP[len(GP)-1-i])

    return GP_out


def downsample_by_factor(x, factor):
    """
    Downsampling OF by a given factor (used to simplify the following LK-method)

    Args:
        x: image as numpy array (H x W)
        factor: factor for downsampling
    Returns:
        downsampled image as numpy array (H/factor x W/factor)
    """

    # procedure taken from assignment 2 problem 1:
    # initialize new downsampled image matrix
    H_new = int(x.shape[0] / factor)
    W_new = int(x.shape[1] / factor)
    downsample = np.zeros((H_new, W_new))
    for i in range(H_new):
        for j in range(W_new):
            # only take every second pixel value and ignore others
            downsample[i, j] = x[i * factor, j * factor]

    return downsample / factor


def upscale_by_factor(of, shape, factor):
    """
    Upscaling of OF to given shape (used to simplify the following LK-method)

    Args:
        x: image as numpy array (H x W)
        shape: new shape (H' x W') for upscaled OF
        factor: upscaling factor
    Returns:
        upscaled image as numpy array (shape)
    """

    # upscale the flow field by factor
    of *= factor
    # then upscale size of the flowfield by griddata interpolation
    x, y = np.meshgrid(np.arange(of.shape[1]), np.arange(of.shape[0]))
    x *= 2
    y *= 2
    points = np.stack([x.flatten(), y.flatten()], axis=-1)
    grid1, grid2 = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    of = griddata(points, of.flatten(), (grid1, grid2), method="linear", fill_value=0)

    return of


###############################
# Coarse-to-fine Lucas-Kanade #
###############################

def coarse_to_fine(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.

    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations

    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape

    # initialize
    im1_temp = pyramid1[-1]
    u = np.zeros(im1_temp.shape)
    v = np.zeros(im1_temp.shape)

    # iterate coarse to fine approach n-times
    for iter in range(n_iter):

        # obtain current (total) OF (original image scale) and scale down to lowest pyramid level
        factor = (len(pyramid1)-1)*2
        u_curr = downsample_by_factor(u, factor)
        v_curr = downsample_by_factor(v, factor)

        # pre warp image 1
        im1_temp = warp(pyramid1[0], u_curr, v_curr)

        # OF per iter (OF change obtained in this iter)
        u_iter = np.zeros(im1_temp.shape)
        v_iter = np.zeros(im1_temp.shape)

        # go through all pyramid levels starting from the coarsest (lowest resolution)
        for p in range(len(pyramid1)):
            # compute derivatives
            Ix, Iy, It = compute_derivatives(im1_temp, pyramid2[p])
            # compute incremental motion
            u_incr, v_incr = compute_motion(Ix, Iy, It, aggregate='gaussian', sigma=2)  # , aggregate='gaussian'
            # add incremental flow to current OF
            u_curr += u_incr
            v_curr += v_incr
            # add incremental flow to OF per iter
            u_iter += u_incr
            v_iter += v_incr

            if p < len(pyramid1) - 1:
                # pre-warp image 1 of next scale with current OF
                # therefore scale the current OF to the next scale
                u_curr = upscale_by_factor(u_curr, shape=pyramid1[p+1].shape, factor=2)
                v_curr = upscale_by_factor(v_curr, shape=pyramid1[p+1].shape, factor=2)
                im1_temp = warp(pyramid1[p+1], u_curr, v_curr)
                # scale of per iter to next scale
                u_iter = upscale_by_factor(u_iter, shape=pyramid1[p+1].shape, factor=2)
                v_iter = upscale_by_factor(v_iter, shape=pyramid1[p+1].shape, factor=2)

        # add flow obtained in this iter to the overall estimate
        u += u_iter
        v += v_iter
        # compute statistics
        cost = compute_cost(warp(pyramid1[-1], u, v), pyramid2[-1])
        print("Cost in iteration ", iter, " is: ", cost)

    assert u.shape == im1.shape and \
           v.shape == im1.shape
    return u, v


###############################
# Coarse-to-fine Lucas-Kanade #
# using image pyramide as     #
# outer loop as described in  #
# the lecture                 #
###############################

def coarse_to_fine2(im1, im2, pyramid1, pyramid2, n_iter=3):
    """Implementation of coarse-to-fine strategy
    for optical flow estimation.
    
    Args:
        im1, im2: first and second image
        pyramid1, pyramid2: Gaussian pyramids corresponding to im1 and im2
        n_iter: number of refinement iterations
    
    Returns:
        u: OF in x direction
        v: OF in y direction
    """
    assert im1.shape == im2.shape

    # initialize
    im1_temp = pyramid1[0]
    u = np.zeros(im1_temp.shape)
    v = np.zeros(im1_temp.shape)

    # go through all pyramid levels
    # outer loop is pyramid iterations as explained in lecture 11
    for p in range(len(pyramid1)):

        for i in range(n_iter):
            # compute derivatives
            Ix, Iy, It = compute_derivatives(im1_temp, pyramid2[p])
            # compute incremental motion
            u_incr, v_incr = compute_motion(Ix, Iy, It, aggregate='gaussian') # , aggregate='gaussian'
            # add increments to motion
            u = np.add(u, u_incr)
            v = np.add(v, v_incr)
            # warp first image towards second and take for next iteration
            im1_temp = warp(pyramid1[p], u, v)
            """
            plt.imshow(pyramid1[p])
            plt.show()
            plt.imshow(im2_temp)
            plt.show()
            plt.imshow(pyramid2[p])
            plt.show()
            # stacking for visualisation
            of = np.stack([u_incr, v_incr], axis=-1)
            # convert to RGB using wheel colour coding
            rgb_image = flow_to_color(of, clip_flow=5)
            # display
            fig = plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
            (ax1, ax2) = fig.subplots(1, 2)
            ax1.imshow(im2_temp)
            ax2.imshow(rgb_image)
            plt.show()
            """
            print(compute_cost(im1_temp, pyramid2[p]))

        # pre-warp image 1 of next scale
        if (p < len(pyramid1)-1):
            # upscale the flow field by factor of 2
            u *= 2
            v *= 2
            # then upscale size of the flowfield by griddata interpolation
            x, y = np.meshgrid(np.arange(u.shape[1]), np.arange(u.shape[0]))
            x *= 2
            y *= 2
            points = np.stack([x.flatten(), y.flatten()], axis=-1)
            grid1, grid2 = np.meshgrid(np.arange(pyramid1[p+1].shape[1]), np.arange(pyramid1[p+1].shape[0]))
            u = griddata(points, u.flatten(), (grid1, grid2), method="linear", fill_value=0)
            v = griddata(points, v.flatten(), (grid1, grid2), method="linear", fill_value=0)
            im1_temp = warp(pyramid1[p+1], u, v)

    assert u.shape == im1.shape and \
            v.shape == im1.shape
    return u, v
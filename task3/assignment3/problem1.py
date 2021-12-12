import numpy as np
import scipy.ndimage
from scipy.ndimage import convolve, maximum_filter


def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    img = np.array(img, dtype=float)
    img = scipy.ndimage.convolve(img, gauss, mode='mirror')
    f2x = np.array([[1, -2, 1]])
    f2y = f2x.T
    #I_xx = scipy.ndimage.convolve(img, f2x, mode='mirror')
    #I_yy = scipy.ndimage.convolve(img, f2y, mode='mirror')
    I_xx = scipy.ndimage.convolve(img, fx, mode='mirror')
    I_xx = scipy.ndimage.convolve(img, fx, mode='mirror')
    I_yy = scipy.ndimage.convolve(img, fy, mode='mirror')
    I_yy = scipy.ndimage.convolve(img, fy, mode='mirror')
    I_xy = scipy.ndimage.convolve(img, fx, mode='mirror')
    I_xy = scipy.ndimage.convolve(I_xy, fy, mode='mirror')


    return I_xx, I_yy, I_xy



def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """

    I_xx = np.array(I_xx, dtype=float)
    I_yy = np.array(I_yy, dtype=float)
    I_xy = np.array(I_xy, dtype=float)
    return sigma**4 * (I_xx * I_yy - np.square(I_xy))



def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """

    criterion = np.array(criterion, dtype=float)
    img = scipy.ndimage.maximum_filter(criterion, size=(5,5), mode='mirror')
    rows = []
    cols = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == criterion[i,j] and img[i,j] > threshold:
                rows.append(i)
                cols.append(j)



    return np.array(rows), np.array(cols)





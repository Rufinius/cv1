import numpy as np
from scipy.ndimage import convolve


def loaddata(path):
    """ Load bayerdata from file

    Args:
        Path of the .npy file
    Returns:
        Bayer data as numpy array (H,W)
    """

    # same as before in problem 1
    # loads an image from .npy file with the given path using numpy.load
    img = np.load(path)
    return img


def separatechannels(bayerdata):
    """ Separate bayer data into RGB channels so that
    each color channel retains only the respective
    values given by the bayer pattern and missing values
    are filled with zero

    Args:
        Numpy array containing bayer data (H,W)
    Returns:
        red, green, and blue channel as numpy array (H,W)
    """

    # set up three color channels with the dimension of the bayerdata
    r = np.zeros(bayerdata.shape)
    g = np.zeros(bayerdata.shape)
    b = np.zeros(bayerdata.shape)

    # iterate through array and write values to the right color channels (following the bayer RGB pattern)
    for i in range(bayerdata.shape[0]):
        for j in range(bayerdata.shape[1]):
            # red color channel
            if i % 2 == 0 and (j+1) % 2 == 0:
                r[i, j] = bayerdata[i, j]
            # green color channel
            elif (i+j) % 2 == 0:
                g[i, j] = bayerdata[i, j]
            # blue color channel
            elif (i+1) % 2 == 0 and j % 2 == 0:
                b[i, j] = bayerdata[i, j]

    return r, g, b


def assembleimage(r, g, b):
    """ Assemble separate channels into image

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Image as numpy array (H,W,3)
    """

    # assembles the separate color channels into one numpy array using numpy.stack
    return np.stack((r, g, b), axis=2)


def interpolate(r, g, b):
    """ Interpolate missing values in the bayer pattern
    by using bilinear interpolation

    Args:
        red, green, blue color channels as numpy array (H,W)
    Returns:
        Interpolated image as numpy array (H,W,3)
    """

    # perform bilinear interpolation for each of the three channels
    # red channel
    r_interpolated = convolve(r, weights=np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]]),
                              mode='mirror')
    # green channel
    g_interpolated = convolve(g, weights=np.array([[0, 1/4, 0], [1/4, 1, 1/4], [0, 1/4, 0]]),
                              mode='mirror')
    # blue channel
    b_interpolated = convolve(b, weights=np.array([[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]]),
                              mode='mirror')

    # assembles the image using the assembleimage function
    return assembleimage(r_interpolated, g_interpolated, b_interpolated)

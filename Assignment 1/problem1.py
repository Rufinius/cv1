import numpy as np
from matplotlib import pyplot as plt


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    # plot the image using matplotlib.pyplot.imshow
    plt.imshow(img)
    # getting rid of the axis ticks as they are not needed for the picture plot
    plt.xticks([])
    plt.yticks([])
    plt.show()


def save_as_npy(path, img):
    """ Save the image array as a .npy file:

    Args:
        Path of the .npy file
        Image as numpy array (H,W,3)
    """

    # saves a given image as .npy file to the given path using numpy.save
    np.save(path, img)


def load_npy(path):
    """ Load and return the .npy file:

    Args:
        Path of the .npy file
    Returns:
        Image as numpy array (H,W,3)
    """

    # loads an image from .npy file with the given path using numpy.load
    img = np.load(path)
    return img


def mirror_horizontal(img):
    """ Create and return a horizontally mirrored image:

    Args:
        Loaded image as numpy array (H,W,3)

    Returns:
        A horizontally mirrored numpy array (H,W,3).
    """

    # mirror the array horizontally along second dimension (W) using numpy.flip
    img_flipped = np.flip(img, axis=1)
    return img_flipped


def display_images(img1, img2):
    """ display the normal and the mirrored image in one plot:

    Args:
        Two image numpy arrays
    """

    # create diagram with two subplots using pyplot.subplot

    plt.subplot(1, 2, 1)
    # plot the image using matplotlib.pyplot.imshow
    plt.imshow(img1)
    # getting rid of the axis ticks as they are not needed for the picture plot
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    # plot the image using matplotlib.pyplot.imshow
    plt.imshow(img2)
    # getting rid of the axis ticks as they are not needed for the picture plot
    plt.xticks([])
    plt.yticks([])

    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian

def load_data(path):
    '''
    Load data from folder data, face images are in the folder facial_images, face features are in the folder facial_features.

    Args:
        path: path of folder data

    Returns:
        imgs: list of face images as numpy arrays 
        feats: list of facial features as numpy arrays 
    '''

    imgs = []
    feats = []

    '''
    Used following source here:
    https://pythonguides.com/python-get-all-files-in-directory/
    https://stackoverflow.com/questions/35723865/read-a-pgm-file-in-python
    '''

    # load data from facial images folder
    for root, dirs, files in os.walk(os.path.join(path, "facial_images")):
        for file in files:
            with open(os.path.join(root, file), 'rb') as pgmf:
                im = plt.imread(pgmf)
            imgs.append(im.astype('float64'))

    # load data from facial features folder
    for root, dirs, files in os.walk(os.path.join(path, "facial_features")):
        for file in files:
            with open(os.path.join(root, file), 'rb') as pgmf:
                im = plt.imread(pgmf)
            feats.append(im.astype('float64'))

    return imgs, feats

def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''
    # using two 1d gaussian kernels to get the fsize x fsize kernel
    # (reused code from last assignment)
    line_filter_vert = gaussian(fsize, sigma).reshape((fsize, 1))
    line_filter_horz = gaussian(fsize, sigma).reshape((1, fsize))
    # use dot product to obtain the 2 dimensional filter
    gaussian_filter = line_filter_vert @ line_filter_horz
    # normalize filter
    gaussian_filter *= 1 / np.sum(gaussian_filter)

    return gaussian_filter

def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)
    '''

    # initialize new downsampled image matrix
    H_new = int(x.shape[0]/factor)
    W_new = int(x.shape[1]/factor)
    downsample = np.zeros((H_new, W_new))
    for i in range(H_new):
        for j in range(W_new):
            # only take every second picel value and ignore others
            downsample[i, j] = x[i*factor, j*factor]

    return downsample


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''
    GP = []
    # append the original image as the first level of the gaussian pyramid
    GP.append(img)
    # add the next levels of the pyramid
    for n in range(nlevels-1):
        # first smooth the image using a convolution with the gaussian kernel
        smoothed = convolve2d(GP[-1], gaussian_kernel(fsize, sigma), mode='same', boundary='symm')
        # now downsample the smoothed image to obtain the next level of the gaussian pyramid
        downsampled = downsample_x2(smoothed)
        GP.append(downsampled)

    return GP

def template_distance(v1, v2):
    '''
    Calculates the distance between the two vectors to find a match.
    Browse the course slides for distance measurement methods to implement this function.
    Tips: 
        - Before doing this, let's take a look at the multiple choice questions that follow. 
        - You may need to implement these distance measurement methods to compare which is better.

    Args:
        v1: vector 1
        v2: vector 2

    Returns:
        Distance
    '''

    # using normalized dot product
    # the dot product is large for small distances between v1 and v2
    # therefore take 1 - normalized dot product as a measure of distance
    dot = 1 - np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # SSD
    ssd = np.sum((v1.reshape(v1.size,) - v2.reshape(v2.size,))**2)

    return dot


def sliding_window(img, feat, step=1):
    ''' 
    A sliding window for matching features to windows with SSDs. When a match is found it returns to its location.
    
    Args:
        img: face image as numpy array (H * W)
        feat: facial feature as numpy array (H * W)
        step: stride size to move the window, default is 1
    Returns:
        min_score: distance between feat and window
    '''

    min_score = None
    # flattened feature to compare window to
    flat_feature = feat.flatten()
    i = 0
    # go through all possible window positions in the image
    while (i+feat.shape[0]) <= img.shape[0]:
        j = 0
        while (j+feat.shape[1]) <= img.shape[1]:
            # get window and flatten it
            window = img[i:i+feat.shape[0], j:j+feat.shape[1]]
            window = window.flatten()
            # get distance score between window and feature
            score = template_distance(window, flat_feature)
            # update the minimum score
            if min_score is None:
                min_score = score
            elif min_score > score:
                min_score = score

            j += step
        i += step

    return min_score


class Distance(object):

    # choice of the method
    METHODS = {1: 'Dot Product', 2: 'SSD Matching'}

    # choice of reasoning
    REASONING = {
        1: 'it is more computationally efficient',
        2: 'it is less sensitive to changes in brightness.',
        3: 'it is more robust to additive Gaussian noise',
        4: 'it can be implemented with convolution',
        5: 'All of the above are correct.'
    }

    def answer(self):
        '''Provide your answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of distance.
            - the following integers provide the reasoning for your choice.
        Note that you have to implement your choice in function template_distance

        For example (made up):
            (1, 1) means
            'I will use Dot Product because it is more computationally efficient.'
        '''

        return (1, 2, 4)  # TODO


def find_matching_with_scale(imgs, feats):
    ''' 
    Find face images and facial features that match the scales 
    
    Args:
        imgs: list of face images as numpy arrays
        feats: list of facial features as numpy arrays 
    Returns:
        match: all the found face images and facial features that match the scales: N * (score, g_im, feat)
        score: minimum score between face image and facial feature
        g_im: face image with corresponding scale
        feat: facial feature
    '''
    match = []
    #(score, g_im, feat) = (None, None, None)

    # go through all features and find the best matching image
    for feat in feats:
        score = None
        g_im = None
        # go through all images and scaled images
        for img in imgs:
            # create the gaussian pyramid for a given image
            GP = gaussian_pyramid(img, 3, 5, 1.4)
            for scaled_img in GP:
                # make sure that feature is at least as big as the scaled_image
                if scaled_img.shape[0] < feat.shape[0] or scaled_img.shape[1] < feat.shape[1]:
                    break
                # get the lowest distance score for the scaled image and the feature
                tmp = sliding_window(scaled_img, feat)
                # update the overall lowest score
                if score is None:
                    score = tmp
                    g_im = scaled_img
                else:
                    if score > tmp:
                        score = tmp
                        g_im = scaled_img
        match.append((score, g_im, feat))

    return match

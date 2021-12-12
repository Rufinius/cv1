import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt

#
# Task 1
#
def load_faces(path, ext=".pgm"):
    """Load faces into an array (N, M),
    where N is the number of face images and
    d is the dimensionality (height*width for greyscale).
    
    Hint: os.walk() supports recursive listing of files 
    and directories in a path
    
    Args:
        path: path to the directory with face images
        ext: extension of the image files (you can assume .pgm only)
    
    Returns:
        x: (N, M) array
        hw: tuple with two elements (height, width)
    """
    x = None
    height = None
    width = None
    # load data from facial images folder
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'rb') as pgmf:
                im = plt.imread(pgmf)
            if x is None:
                # set the initial values and also height and width
                x = np.array(im)
                height = x.shape[0]
                width = x.shape[1]
                x = x.flatten().reshape(1, height*width)
            else:
                # append new image to array x
                im_new = np.array(im).flatten().reshape(1, height*width)
                x = np.append(x, im_new, axis=0)
    
    return x, (height, width)

#
# Task 2
#

"""
This is a multiple-choice question
"""

class PCA(object):

    # choice of the method
    METHODS = {
                1: "SVD",
                2: "Eigendecomposition"
    }

    # choice of reasoning
    REASONING = {
                1: "it can be applied to any matrix and is more numerically stable",
                2: "it is more computationally efficient for our problem",
                3: "it allows to compute eigenvectors and eigenvalues of any matrix",
                4: "we can find the eigenvalues we need for our problem from the singular values",
                5: "we can find the singular values we need for our problem from the eigenvalues"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple:
            - the first integer is the choice of the method you will use in your implementation of PCA
            - the following integers provide the reasoning for your choice

        For example (made up):
            (2, 1, 5) means
            "I will use eigendecomposition because
                - we can apply it to any matrix
                - we need singular values which we can obtain from the eigenvalues"
        """

        return (1, 1, 2, 4)

#
# Task 3
#

def compute_pca(X):
    """PCA implementation
    
    Args:
        X: (N, M) an array with N M-dimensional features
    
    Returns:
        u: (M, N) bases with principal components
        lmb: (N, ) corresponding variance
    """
    # transpose the array first
    X = X.T

    # calculate the mean and subtract it from the data
    mean = np.mean(X, axis=1).reshape(X.shape[0], 1)
    mean_matrix = mean
    for i in range(1, X.shape[1]):
        mean_matrix = np.append(mean_matrix, mean, axis=1)
    x_centered = X - mean_matrix

    # calculate SVD for the centered data
    u, s, vh = np.linalg.svd(x_centered)

    # calculate the eigenvalues from the singular values
    lmb = np.power(s, 2)/x_centered.shape[1]

    return u, lmb

#
# Task 4
#

def basis(u, s, p = 0.5):
    """Return the minimum number of basis vectors 
    from matrix U such that they account for at least p percent
    of total variance.
    
    Hint: Do the singular values really represent the variance?
    
    Args:
        u: (M, M) contains principal components.
        For example, i-th vector is u[:, i]
        s: (M, ) variance along the principal components.
    
    Returns:
        v: (M, D) contains M principal components from N
        containing at most p (percentile) of the variance.
    
    """
    # calculate the total variance
    total_variance = np.sum(s)

    var_counted = 0
    i = 0
    # add values to the variance_counter until the share is greater than p
    while var_counted/total_variance < p:
        var_counted += s[i]
        i += 1

    # return only the first D=i-1 eigenvectors
    return u[:, :i]

#
# Task 5
#
def project(face_image, u):
    """Project face image to a number of principal
    components specified by num_components.
    
    Args:
        face_image: (N, ) vector (N=h*w) of the face
        u: (N,M) matrix containing M principal components. 
        For example, (:, 1) is the second component vector.
    
    Returns:
        image_out: (N, ) vector, projection of face_image on 
        principal components
    """

    # project image using the M principal components stored in u
    projected_image = np.dot(np.transpose(u), face_image.reshape(face_image.size, 1))

    # project image back from the principal components
    image_out = np.dot(u, projected_image)

    return image_out

#
# Task 6
#

"""
This is a multiple-choice question
"""
class NumberOfComponents(object):

    # choice of the method
    OBSERVATION = {
                1: "The more principal components we use, the sharper is the image",
                2: "The fewer principal components we use, the smaller is the re-projection error",
                3: "The first principal components mostly correspond to local features, e.g. nose, mouth, eyes",
                4: "The first principal components predominantly contain global structure, e.g. complete face",
                5: "The variations in the last principal components are perceptually insignificant; these bases can be neglected in the projection"
    }

    def answer(self):
        """Provide answer in the return value.
        This function returns one tuple describing you observations

        For example: (1, 3)
        """

        return (1, 4, 5)


#
# Task 7
#
def search(Y, x, u, top_n):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        Y: (N, M) centered array with N d-dimensional features
        x: (1, M) image we would like to retrieve
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        top_n: integer, return top_n closest images in L2 sense.
    
    Returns:
        Y: (top_n, M)
    """

    top_images = None
    scorelist = []
    # project the search image to the eigenvector space
    img_query = np.dot(np.transpose(u), x.reshape(x.size, 1))
    # iterate through all images
    for i in range(Y.shape[0]):
        img_value = np.dot(np.transpose(u), Y[i].reshape(Y[i].size, 1))
        # l2 distance score
        score = np.linalg.norm(img_query-img_value)**2
        # append tupel of score and image id to scorelist
        scorelist.append((score, i))

    # sort scorelist and find the top n values smallest values because smallest distances
    scorelist.sort(key=lambda t: t[0])
    for i in range(top_n):
        # append images to result vector
        if top_images is None:
            top_images = Y[scorelist[i][1]].reshape(1, Y.shape[1])
        else:
            top_images = np.append(top_images, Y[scorelist[i][1]].reshape(1, Y.shape[1]), axis=0)

    return top_images

#
# Task 8
#
def interpolate(x1, x2, u, N):
    """Search for the top most similar images
    based on a given number of components in their PCA decomposition.
    
    Args:
        x1: (1, M) array, the first image
        x2: (1, M) array, the second image
        u: (M, D) basis vectors. Note, we already assume D has been selected.
        N: number of interpolation steps (including x1 and x2)

    Hint: you can use np.linspace to generate N equally-spaced points on a line
    
    Returns:
        Y: (N, M) interpolated results. The first dimension is in the index into corresponding
        image; Y[0] == project(x1, u); Y[-1] == project(x2, u)
    """
    # project images x1 and x2 to eigenvector space
    projection1 = np.dot(np.transpose(u), x1.reshape(x1.size, 1))
    projection2 = np.dot(np.transpose(u), x2.reshape(x2.size, 1))

    # perform interpolation in pca space
    interpolation = None
    for i in range(projection1.shape[0]):
        # use linspace for the interpolation of every element of the projected vectors
        interpolated = np.linspace(projection1[i], projection2[i], N)
        if interpolation is None:
            interpolation = interpolated.reshape(N, 1)
        else:
            interpolation = np.append(interpolation, interpolated.reshape(N, 1), axis=1)

    # project from pca space back to images
    interpolated_images = None
    for i in range(N):
        img_vector = np.zeros(u.shape[0])
        # calculate the projection back to image space
        for d in range(u.shape[1]):
            img_vector += interpolation[i][d] * u[:, d]
        if interpolated_images is None:
            interpolated_images = img_vector.reshape(1, img_vector.size)
        else:
            interpolated_images = np.append(interpolated_images, img_vector.reshape(1, img_vector.size), axis=0)

    return interpolated_images


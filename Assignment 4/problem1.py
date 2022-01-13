import matplotlib.pyplot as plt
import numpy as np


def condition_points(points):
    """ Conditioning: Normalization of coordinates for numeric stability 
    by substracting the mean and dividing by half of the component-wise
    maximum absolute value.
    Args:
        points: (l, 3) numpy array containing unnormalized homogeneous coordinates.

    Returns:
        ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
        T: (3, 3) numpy array, transformation matrix for conditioning
    """
    t = np.mean(points, axis=0)[:-1]
    s = 0.5 * np.max(np.abs(points), axis=0)[:-1]
    T = np.eye(3)
    T[0:2,2] = -t
    T[0:2, 0:3] = T[0:2, 0:3] / np.expand_dims(s, axis=1)
    ps = points @ T.T
    return ps, T


def enforce_rank2(A):
    """ Enforces rank 2 to a given 3 x 3 matrix by setting the smallest
    eigenvalue to zero.
    Args:
        A: (3, 3) numpy array, input matrix

    Returns:
        A_hat: (3, 3) numpy array, matrix with rank at most 2
    """

    # following the procedure from lecture 9 slide 32

    # perform SVD on preliminary fundamental matrix A
    U, s, Vh = np.linalg.svd(A)
    # set smallest eigenvale to 0
    D = np.diag(s)
    D[2, 2] = 0
    # recompute fundamental matrix
    A_hat = U @ D @ Vh

    return A_hat


def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    # first build matrix A by stacking point correspondences using the formula from lecture 9 slide 30
    A = None
    for n in range(p1.shape[0]):
        x = p2[n, 0]/p2[n, 2]
        y = p2[n, 1]/p2[n, 2]
        xd = p1[n, 0]/p1[n, 2]
        yd = p1[n, 1]/p1[n, 2]
        line = np.array([[x*xd, y*xd, xd, x*yd, y*yd, yd, x, y, 1]])
        if A is None:
            A = line
        else:
            A = np.append(A, line, axis=0)

    # perform SVD
    u, s, vh = np.linalg.svd(A)
    # solution is the last right singular vector
    f = vh[-1]
    # reshape into fundamental matrix
    F = f.reshape((3, 3), order='C')

    # enfore rank 2
    F = enforce_rank2(F)

    return F



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    # condition points
    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)

    # compute fundamental matrix (conditioned)
    F = compute_fundamental(ps1, ps2)

    # uncondition fundamental matrix
    F = T1.T @ F @ T2

    return F


def draw_epipolars(F, p1, img):
    """ Computes the coordinates of the n epipolar lines (X1, Y1) on the left image border and (X2, Y2)
    on the right image border.
    Args:
        F: (3, 3) numpy array, fundamental matrix 
        p1: (n, 2) numpy array, cartesian coordinates of the point correspondences in the image
        img: (H, W, 3) numpy array, image data

    Returns:
        X1, X2, Y1, Y2: (n, ) numpy arrays containing the coordinates of the n epipolar lines
            at the image borders
    """
    # initialize coordinate arrays:
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []

    # shape of image
    H = img.shape[0]
    W = img.shape[1]

    # now iterate through all points p1
    for n in range(p1.shape[0]):
        point = np.append(p1[n], np.ones(1)).reshape(3, 1)
        # find epipolar line:
        l = F @ point
        # normalize l
        l = l / l[2, 0]

        # initialize solution points
        points = None
        # check intersections with x=0, x=W-1, y=0, y=H-1
        # using formula: p.T @ l = 0
        # set one value of border coordinate fixed, calculate the other and see if it is a valid border intersection

        # x=0, y=?
        y_cord = (-0*l[0, 0] - 1*l[2, 0])/l[1, 0]
        if 0 <= y_cord <= H-1:
            points = np.array([[0, y_cord]])
        # x=W-1, y=?
        y_cord = (-(W-1)*l[0, 0] - 1*l[2, 0])/l[1, 0]
        if 0 <= y_cord <= H-1:
            if points is None:
                points = np.array([[W-1, y_cord]])
            else:
                points = np.append(points, np.array([[W-1, y_cord]]), axis=0)
        # y=0, x=?
        x_cord = (- 0 * l[1, 0] - 1 * l[2, 0])/l[0, 0]
        if 0 <= x_cord <= W-1:
            if points is None:
                points = np.array([[x_cord, 0]])
            else:
                points = np.append(points, np.array([[x_cord, 0]]), axis=0)
        # y=H-1, x=?
        x_cord = (- (H-1)*l[1, 0] - 1*l[2, 0])/l[0, 0]
        if 0 <= x_cord <= W-1:
            points = np.append(points, np.array([[x_cord, H-1]]), axis=0)

        # check that exactly two points are found
        assert points.shape[0] == 2

        # find point that is further left (smaller y value)
        if points[0, 1] < points[1, 1]:
            X1.append(points[0, 0])
            Y1.append(points[0, 1])
            X2.append(points[1, 0])
            Y2.append(points[1, 1])
        else:
            X1.append(points[1, 0])
            Y1.append(points[1, 1])
            X2.append(points[0, 0])
            Y2.append(points[0, 1])

    return np.array(X1), np.array(X2), np.array(Y1), np.array(Y2)


def compute_residuals(p1, p2, F):
    """
    Computes the maximum and average absolute residual value of the epipolar constraint equation.
    Args:
        p1: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 1
        p2: (n, 3) numpy array containing the homogeneous correspondence coordinates from image 2
        F:  (3, 3) numpy array, fundamental matrix

    Returns:
        max_residual: maximum absolute residual value
        avg_residual: average absolute residual value
    """

    n = p1.shape[0]
    avg_residual = 0
    max_residual = 0

    # iterate through all points
    for pn in range(n):
        res = np.abs(p1[pn].reshape(1, 3) @ F @ p2[pn].reshape(3, 1))
        avg_residual += res/n
        if max_residual < res:
            max_residual = res

    return max_residual, avg_residual


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    # right nullspace of F.T corresponds to e1
    # right nullspace of F corresponds to e2
    # use SVD to calculate the right nullspace:

    # perform SVD
    u, s, vh = np.linalg.svd(F.T)
    # solution is the last right singular vector
    e1 = vh[-1]
    # to cartesian coordinates
    e1 = np.array([e1[0]/e1[2], e1[1]/e1[2]])

    # perform SVD
    u, s, vh = np.linalg.svd(F)
    # solution is the last right singular vector
    e2 = vh[-1]
    # to cartesian coordinates
    e2 = np.array([e2[0] / e2[2], e2[1] / e2[2]])

    return e1, e2

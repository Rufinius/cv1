import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space


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

    U, D, V = np.linalg.svd(A)
    D[2] = 0
    diag = np.zeros((3,3))
    np.fill_diagonal(diag, D)
    F = U @ diag @ V

    return F


def compute_fundamental(p1, p2):
    """ Computes the fundamental matrix from conditioned coordinates.
    Args:
        p1: (n, 3) numpy array containing the conditioned coordinates in the left image
        p2: (n, 3) numpy array containing the conditioned coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix
    """

    A = np.zeros((p1.shape[0], 9))
    for i in range(p1.shape[0]):
        A[i, 0] = p1[i, 0] * p2[i, 0]
        A[i, 1] = p1[i, 1] * p2[i, 0]
        A[i, 2] = p2[i, 0]
        A[i, 3] = p1[i, 0] * p2[i, 1]
        A[i, 4] = p1[i, 1] * p2[i, 1]
        A[i, 5] = p2[i ,1]
        A[i, 6] = p1[i, 0]
        A[i, 7] = p1[i, 1]
        A[i, 8] = 1

    U, S, V = np.linalg.svd(A)
    F = V[-1]
    F = F.reshape((3,3))

    return enforce_rank2(F)



def eight_point(p1, p2):
    """ Computes the fundamental matrix from unconditioned coordinates.
    Conditions coordinates first.
    Args:
        p1: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the left image
        p2: (n, 3) numpy array containing the unconditioned homogeneous coordinates in the right image

    Returns:
        F: (3, 3) numpy array, fundamental matrix with respect to the unconditioned coordinates
    """

    ps1, T1 = condition_points(p1)
    ps2, T2 = condition_points(p2)

    F_cond = compute_fundamental(ps1, ps2)

    return T2.T @ F_cond @ T1




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

    p_h = np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1)
    lines = []
    X1 = [0]*p_h.shape[0]
    Y1 = []
    X2 = [img.shape[1]]*p_h.shape[0]
    Y2 = []
    for i in range(p_h.shape[0]):
        lines.append(F @ p_h[i])
        # lines[i][0] = lines[i][0] / lines[i][2]
        # lines[i][1] = lines[i][1] / lines[i][2]
        # lines[i][2] = lines[i][2] / lines[i][2]
        Y1.append(- (lines[i][2]/lines[i][1]))
        Y2.append(- (lines[i][0]/lines[i][1])*X2[i] - (lines[i][2]/lines[i][1]))

    return X1, X2, Y1, Y2


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

    maxi = 0
    mean = 0

    for i in range(p1.shape[0]):
        tmp = np.abs(p1[i].T @ F @ p2[i])
        if tmp > maxi:
            maxi = tmp
        mean += tmp

    mean /= p1.shape[0]

    return maxi, mean


def compute_epipoles(F):
    """ Computes the cartesian coordinates of the epipoles e1 and e2 in image 1 and 2 respectively.
    Args:
        F: (3, 3) numpy array, fundamental matrix

    Returns:
        e1: (2, ) numpy array, cartesian coordinates of the epipole in image 1
        e2: (2, ) numpy array, cartesian coordinates of the epipole in image 2
    """

    ns1 = null_space(F.T)
    ns2 = null_space(F)

    e1 = np.array([ns1[0]/ns1[2], ns1[1]/ns1[2]])
    e2 = np.array([ns2[0]/ns2[2], ns2[1]/ns2[2]])

    return e1, e2

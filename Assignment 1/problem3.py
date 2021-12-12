import numpy as np
import scipy.linalg


def load_points(path):
    '''
    Load points from path pointing to a numpy binary file (.npy). 
    Image points are saved in 'image'
    Object points are saved in 'world'

    Returns:
        image: A Nx2 array of 2D points form image coordinate 
        world: A N*3 array of 3D points form world coordinate
    '''

    # assuming the return values should actually be Nx3 and Nx4 respectively:

    # load data using numpy.load
    data_array = np.load(path)
    # retrieve data as numpy arrays
    image_homogeneous = data_array['image']
    world_homogeneous = data_array['world']

    """
    # converting from homogeneous coordinates
    # getting rid of last dimension: here we could just delete it because it is always zero
    # but used more general formula from l2-image_formation slide 29
    image = np.zeros((image_homogeneous.shape[0], 2))
    for i in range(image_homogeneous.shape[0]):
        image[i, 0] = image_homogeneous[i, 0] / image_homogeneous[i, 2]
        image[i, 1] = image_homogeneous[i, 1] / image_homogeneous[i, 2]
    world = np.zeros((world_homogeneous.shape[0], 3))
    for i in range(world_homogeneous.shape[0]):
        world[i, 0] = world_homogeneous[i, 0] / world_homogeneous[i, 3]
        world[i, 1] = world_homogeneous[i, 1] / world_homogeneous[i, 3]
        world[i, 2] = world_homogeneous[i, 2] / world_homogeneous[i, 3]
    return image, world
    """
    return image_homogeneous, world_homogeneous


def create_A(x, X):
    """Creates (2*N, 12) matrix A from 2D/3D correspondences
    that comes from cross-product
    
    Args:
        x and X: N 2D and 3D point correspondences (homogeneous)
        
    Returns:
        A: (2*N, 12) matrix A
    """

    N, _ = x.shape
    assert N == X.shape[0]

    # create an zero matrix of the required shape
    A = np.zeros((2*N, 12))
    # fill with the correct values according to l2-image_formation slide 51
    for n in range(N):
        A[2*n, 4:8] = -1 * X[n]
        A[2*n, 8:] = x[n, 1] * X[n]
        A[2*n+1, :4] = X[n]
        A[2*n+1, 8:] = -x[n, 0] * X[n]
    return A


def homogeneous_Ax(A):
    """Solve homogeneous least squares problem (Ax = 0, s.t. norm(x) == 0),
    using SVD decomposition as in the lecture.

    Args:
        A: (2*N, 12) matrix A
    
    Returns:
        P: (3, 4) projection matrix P
    """
    # assuming norm(x) == 1 is the correct version
    U, S, Vh = np.linalg.svd(A)
    v12 = Vh[-1]
    P = v12.reshape((3, 4))

    return P


def solve_KR(P):
    """Using th RQ-decomposition find K and R 
    from the projection matrix P.
    Hint 1: you might find scipy.linalg useful here.
    Hint 2: recall that K has 1 in the the bottom right corner.
    Hint 3: RQ decomposition is not unique (up to a column sign).
    Ensure positive element in K by inverting the sign in K columns 
    and doing so correspondingly in R.

    Args:
        P: 3x4 projection matrix.
    
    Returns:
        K: 3x3 matrix with intrinsics
        R: 3x3 rotation matrix 
    """
    # computing the RQ decomposition
    K, R = scipy.linalg.rq(P[:3, :3])

    if K[0, 0] < 0:
        # ensure that focal length is positive by changing the sign of the first column of K
        K[:, 0] = -1 * K[:, 0]
        # change the sign of the first row of R so that this counteracts the change in the first column K
        R[0, :] = -1 * R[0, :]

    if K[1, 1] < 0:
        # ensure that focal length is positive by changing the sign of the second column of K
        K[:, 1] = -1 * K[:, 1]
        # change the sign of the second row of R so that this counteracts the change in the second column K
        R[1, :] = -1 * R[1, :]

    # ensure that bottom right corner of K is 1 by dividing all elements of K by that value
    value = K[2, 2]
    K = K / value
    # change R so that this counteracts the change of K
    R = R * value

    return K, R


def solve_c(P):
    """Find the camera center coordinate from P
    by finding the nullspace of P with SVD.

    Args:
        P: 3x4 projection matrix
    
    Returns:
        c: 3x1 camera center coordinate in the world frame
    """
    # perform SVD and take the eigenvector corresponding to the smalles eigenvalue
    U, S, Vh = np.linalg.svd(P)
    v12 = Vh[-1]
    # transform from homogeneous coordinates (by dividing through the last entry)
    c = np.zeros((3, 1))
    for i in range(3):
        c[i] = v12[i] / v12[3]

    return c

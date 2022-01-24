import cv2
import numpy as np


def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    summ = 0
    for i in range(patch1.shape[0]):
        for j in range(patch1.shape[1]):
            summ += ((patch1[i, j] - patch2[i, j])**2)

    assert np.isscalar(summ)
    # assert cost_ssd > 0
    return summ


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """

    mean_p1 = np.mean(patch1)
    mean_p2 = np.mean(patch2)

    cost_nc_1 = ((patch1.reshape((121, 1)) - mean_p1).T @ (patch2.reshape((121, 1)) - mean_p2))
    cost_nc_2 = (np.linalg.norm(patch1.reshape((121, 1)) - mean_p1, ord=1) * np.linalg.norm(patch2.reshape((121, 1)) - mean_p2, ord=1))

    cost_nc_compl = float(cost_nc_1 / cost_nc_2)

    assert np.isscalar(cost_nc_compl)
    # assert cost_nc > 0
    return cost_nc_compl


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        input_disparity: input disparity as an integer value        
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    ssd = cost_ssd(patch1, patch2)
    nc = cost_nc(patch1, patch2)
    cost_val = (1/(patch1.shape[0]**2)) * ssd + alpha * nc
    
    assert np.isscalar(cost_val)
    return cost_val


def pad_image(input_img, window_size, padding_mode='symmetric'):
    """Output the padded image
    
    Args:
        input_img: an input image as a numpy array
        window_size: the window size as a scalar value, odd number
        padding_mode: the type of padding scheme, among 'symmetric', 'reflect', or 'constant'
        
    Returns:
        padded_img: padded image as a numpy array of the same type as image
    """
    assert np.isscalar(window_size)
    assert window_size % 2 == 1

    if padding_mode == 'symmetric':
        padded_img = np.pad(input_img, int(window_size/2), mode='symmetric')
    elif padding_mode == 'constant':
        padded_img = np.pad(input_img, int(window_size/2), mode='constant', constant_values=0)
    elif padding_mode == 'reflect':
        padded_img = np.pad(input_img, int(window_size/2), mode='reflect')
    else:
        print('ERROR: specified padding mode not known used constant instead')
        padded_img = np.pad(input_img, int(window_size / 2), mode='constant', constant_values=0)


    return padded_img


def compute_disparity(padded_img_l, padded_img_r, max_disp, window_size, alpha):
    """Compute the disparity map by using the window-based matching:    
    
    Args:
        padded_img_l: The padded left-view input image as 2-dimensional (H,W) numpy array
        padded_img_r: The padded right-view input image as 2-dimensional (H,W) numpy array
        max_disp: the maximum disparity as a search range
        window_size: the patch size for window-based matching, odd number
        alpha: the weighting parameter for the cost function
    Returns:
        disparity: numpy array (H,W) of the same type as image
    """

    assert padded_img_l.ndim == 2 
    assert padded_img_r.ndim == 2 
    assert padded_img_l.shape == padded_img_r.shape
    assert max_disp > 0
    assert window_size % 2 == 1

    disparity = np.zeros((padded_img_l.shape[0]-window_size+1, padded_img_l.shape[1]-window_size+1))

    for i in range(padded_img_l.shape[0] - window_size + 1):
        for j in range(padded_img_l.shape[1] - window_size + 1):
            min_cost = float('inf')
            for k in range(max_disp):
                if j+k+window_size > 265 or i+window_size > 118:
                    break
                cost = cost_function(padded_img_r[i:i+window_size:, j:j+window_size:],
                                     padded_img_l[i:i+window_size:, j+k:j+k+window_size:], alpha)
                if cost < min_cost:
                    min_cost = cost
            disparity[i, j] = min_cost
            min_cost = float('inf')


    assert disparity.ndim == 2
    return disparity

def compute_aepe(disparity_gt, disparity_res):
    """Compute the average end-point error of the estimated disparity map:
    
    Args:
        disparity_gt: the ground truth of disparity map as (H, W) numpy array
        disparity_res: the estimated disparity map as (H, W) numpy array
    
    Returns:
        aepe: the average end-point error as a floating point value
    """
    assert disparity_gt.ndim == 2 
    assert disparity_res.ndim == 2 
    assert disparity_gt.shape == disparity_res.shape

    disparity_gt = disparity_gt.reshape((disparity_gt.shape[0]*disparity_gt.shape[1], 1))
    disparity_res = disparity_res.reshape((disparity_res.shape[0]*disparity_res.shape[1], 1))

    aepe = 1/disparity_gt.shape[0]*disparity_gt.shape[1] * np.linalg.norm(disparity_gt-disparity_res, ord=1)


    assert np.isscalar(aepe)
    return aepe

def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    return alpha


"""
This is a multiple-choice question
"""
class WindowBasedDisparityMatching(object):

    def answer(self):
        """Complete the following sentence by choosing the most appropriate answer 
        and return the value as a tuple.
        (Element with index 0 corresponds to Q1, with index 1 to Q2 and so on.)
        
        Q1. [?] is better for estimating disparity values on sharp objects and object boundaries
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)
        
        Q2. [?] is good for estimating disparity values on locally non-textured area.
          1: Using a smaller window size (e.g., 3x3)
          2: Using a bigger window size (e.g., 11x11)

        Q3. When using a [?] padding scheme, the artifacts on the right border of the estimated disparity map become the worst.
          1: constant
          2: reflect
          3: symmetric

        Q4. The inaccurate disparity estimation on the left image border happens due to [?].
          1: the inappropriate padding scheme
          2: the absence of corresponding pixels
          3: the limitations of the fixed window size
          4: the lack of global information

        """

        return (-1, -1, -1, -1)

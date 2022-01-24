import numpy as np


def cost_ssd(patch1, patch2):
    """Compute the Sum of Squared Pixel Differences (SSD):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_ssd: the calcuated SSD cost as a floating point value
    """

    cost_ssd = 0
    for m1 in range(patch1.shape[0]):
        for m2 in range(patch1.shape[1]):
            cost_ssd += (patch1[m1, m2] - patch2[m1, m2])**2

    assert np.isscalar(cost_ssd)
    return cost_ssd


def cost_nc(patch1, patch2):
    """Compute the normalized correlation cost (NC):
    
    Args:
        patch1: input patch 1 as (m, m, 1) numpy array
        patch2: input patch 2 as (m, m, 1) numpy array
    
    Returns:
        cost_nc: the calcuated NC cost as a floating point value
    """
    patch1 = patch1.flatten()
    patch2 = patch2.flatten()
    patch1_c = (patch1 - np.mean(patch1)).reshape(patch1.shape[0], 1)
    patch2_c = (patch2 - np.mean(patch2)).reshape(patch2.shape[0], 1)

    # using 1-norm as stated in the discussion forum
    cost_nc = (patch1_c.T @ patch2_c)/(np.linalg.norm(patch1_c, ord=1) * np.linalg.norm(patch2_c, ord=1))
    cost_nc = cost_nc[0, 0]

    assert np.isscalar(cost_nc)
    return cost_nc


def cost_function(patch1, patch2, alpha):
    """Compute the cost between two input window patches given the disparity:
    
    Args:
        patch1: input patch 1 as (m, m) numpy array
        patch2: input patch 2 as (m, m) numpy array
        alpha: the weighting parameter for the cost function
    Returns:
        cost_val: the calculated cost value as a floating point value
    """
    assert patch1.shape == patch2.shape 

    cost_val = 1/(patch1.shape[0]**2) * cost_ssd(patch1, patch2) + alpha * cost_nc(patch1, patch2)
    
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

    if padding_mode == 'constant':
        padded_img = np.pad(array=input_img, pad_width=int(window_size/2), mode=padding_mode, constant_values=0)
    else:
        padded_img = np.pad(array=input_img, pad_width=int(window_size/2), mode=padding_mode)

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

    # initialize disparity
    disparity = np.zeros(shape=(padded_img_l.shape[0]-window_size+1, padded_img_l.shape[1]-window_size+1))

    # iterate through every possible patch window in the left image
    for yl in range(padded_img_l.shape[0]-window_size+1):
        for xl in range(padded_img_l.shape[1]-window_size+1):
            patch1 = padded_img_l[yl:yl+window_size, xl:xl+window_size]

            # go through candidate disparity values for the second patch
            min_cost = np.infty
            best_disp = 0
            for disp in range(min(max_disp, xl)+1):
                patch2 = padded_img_r[yl:yl+window_size, xl-disp:xl+window_size-disp]
                cost = cost_function(patch1, patch2, alpha)
                if cost < min_cost:
                    min_cost = cost
                    best_disp = disp

            # write found disparity value into disparity matrix
            disparity[yl, xl] = best_disp

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

    N = disparity_gt.shape[0] * disparity_gt.shape[1]
    aepe = 0
    for y in range(disparity_gt.shape[0]):
        for x in range(disparity_gt.shape[1]):
            aepe += 1/N * (np.abs(disparity_gt[y, x] - disparity_res[y, x]))

    assert np.isscalar(aepe)
    return aepe


def optimal_alpha():
    """Return alpha that leads to the smallest EPE 
    (w.r.t. other values)"""
    
    #
    # Fix alpha
    #
    alpha = np.random.choice([-0.06, -0.01, 0.04, 0.1])
    return -0.01


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

        return (1, 2, 1, 2)

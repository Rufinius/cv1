import math
import numpy as np
from scipy import ndimage
from scipy.signal.windows import gaussian


def gauss2d(sigma, fsize):
  """
  Args:
    sigma: width of the Gaussian filter
    fsize: dimensions of the filter

  Returns:
    g: *normalized* Gaussian filter
  """
  line_filter_vert = gaussian(fsize[0], sigma).reshape((fsize[0],1))
  line_filter_horz = gaussian(fsize[1], sigma).reshape((1, fsize[1]))
  gaussian_filter = line_filter_vert @ line_filter_horz
  gaussian_filter *= 1 / np.sum(gaussian_filter)
  return gaussian_filter


def createfilters():
  """
  Returns:
    fx, fy: filters as described in the problem assignment
  """

  # central differences as described in the task
  central_differences = np.array([[1/2, 0, -1/2]])

  # 1dimensional gaussian filter
  gaussian_filter = gauss2d(0.9, (3, 1))

  # combine both filters to give the filters described in the problem assignment
  fx = gaussian_filter @ central_differences
  fy = central_differences.T @ gaussian_filter.T

  return fx, fy


def filterimage(I, fx, fy):
  """ Filter the image with the filters fx, fy.
  You may use the ndimage.convolve scipy-function.

  Args:
    I: a (H,W) numpy array storing image data
    fx, fy: filters

  Returns:
    Ix, Iy: images filtered by fx and fy respectively
  """

  # use the ndimage.convolve function with the given filters to filter the image
  Ix = ndimage.convolve(I, fx)
  Iy = ndimage.convolve(I, fy)
  return Ix, Iy


def detectedges(Ix, Iy, thr):
  """ Detects edges by applying a threshold on the image gradient magnitude.

  Args:
    Ix, Iy: filtered images
    thr: the threshold value

  Returns:
    edges: (H,W) array that contains the magnitude of the image gradient at edges and 0 otherwise
  """

  # set up a zero matrix for the gradient magnitudes of the edges
  edges = np.zeros(Ix.shape)
  # go through each entry of the given gradients
  for i in range(Ix.shape[0]):
    for j in range(Ix.shape[1]):
      # calculate the magnitude of the gradient for the given position
      mag = np.sqrt(Ix[i, j]**2 + Iy[i, j]**2)
      if mag > thr:
        # if the magnitude is greater than the threshold, the value is written into edges[][]
        edges[i, j] = mag
  return edges


def nonmaxsupp(edges, Ix, Iy):
  """ Performs non-maximum suppression on an edge map.

  Args:
    edges: edge map containing the magnitude of the image gradient at edges and 0 otherwise
    Ix, Iy: filtered images

  Returns:
    edges2: edge map where non-maximum edges are suppressed
  """
  # use edges2 for modification of edges, so that the original edges array is unchanged
  edges2 = np.copy(edges)

  # go through all pixels in edges
  for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
      theta = np.arctan2(Iy[i, j], Ix[i, j]) * 180 / np.pi
      if theta < -90:
        theta += 180
      elif theta > 90:
        theta -= 180

      # handle top-to-bottom edges: theta in [-90, -67.5] or (67.5, 90]
      if -90 <= theta <= -67.5 or 67.5 < theta <= 90:
        # make sure that neighbour is valid (inside the picture)
        # then check if magnitude of neighboring pixel is bigger and if yes remove edge2 value
        if i > 0:
          if edges[i-1, j] > edges[i, j]:
            edges2[i, j] = 0
        if i < edges.shape[0]-1:
          if edges[i+1, j] > edges[i, j]:
            edges2[i, j] = 0

      # handle left-to-right edges: theta in (-22.5, 22.5]
      elif -22.5 < theta <= 22.5:
        # make sure that neighbour is valid (inside the picture)
        # then check if magnitude of neighboring pixel is bigger and if yes remove edge2 value
        if j > 0:
          if edges[i, j-1] > edges[i, j]:
            edges2[i, j] = 0
        if j < edges.shape[1] - 1:
          if edges[i, j+1] > edges[i, j]:
            edges2[i, j] = 0

      # handle bottomleft-to-topright edges: theta in (22.5, 67.5]
      elif 22.5 < theta <= 67.5:
        # make sure that neighbour is valid (inside the picture)
        # then check if magnitude of neighboring pixel is bigger and if yes remove edge2 value
        if i>0 and j>0:
          if edges[i-1, j-1] > edges[i, j]:
            edges2[i, j] = 0
        if i < edges.shape[0]-1 and j < edges.shape[1]-1:
          if edges[i+1, j+1] > edges[i, j]:
            edges2[i, j] = 0

      # handle topleft-to-bottomright edges: theta in [-67.5, -22.5]
      elif -67.5 <= theta <= -22.5:
        # make sure that neighbour is valid (inside the picture)
        # then check if magnitude of neighboring pixel is bigger and if yes remove edge2 value
        if i>0 and j < edges.shape[1] - 1:
          if edges[i-1, j+1] > edges[i, j]:
            edges2[i, j] = 0
        if i < edges.shape[0]-1 and j>0:
          if edges[i+1, j-1] > edges[i, j]:
            edges2[i, j] = 0

      else:
        print("Shouldnt get here -> DEBUG!")

  return edges2

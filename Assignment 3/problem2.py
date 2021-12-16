import numpy as np


class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.

        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image

        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        # initialise distance matrix
        distances = np.zeros((features2.shape[1], features1.shape[1]))
        # iterate through all pairs of features
        for m in range(features1.shape[1]):
            for n in range(features2.shape[1]):
                # compute squared euclidean distance as given in the task description
                distances[n, m] = (np.linalg.norm(features1[:, m] - features2[:, n]))**2
        return distances

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.

        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix

        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        # create array for pairs
        pairs = np.zeros((min(p1.shape[0], p2.shape[0]), 4))

        # assuming that every point in the smaller set has a corresponding
        # interest point in the larger set!

        if (p1.shape[0] < p2.shape[0]):
            # p1 is smaller set
            for m in range(p1.shape[0]):
                # find the nearest neighbour (minimum distance)
                id = np.argmin(distances[:, m])
                pairs[m, :2] = p1[m]
                pairs[m, 2:4] = p2[id]
        else:
            # p1 is larger (or equal sized) set
            for n in range(p2.shape[0]):
                # find the nearest neighbour (minimum distance)
                id = np.argmin(distances[n])
                pairs[n, :2] = p1[id]
                pairs[n, 2:4] = p2[n]
        return pairs

    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.

        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select

        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """
        # create sample1 and sample2 arrays
        sample1 = np.zeros((k, 2))
        sample2 = np.zeros((k, 2))

        # create random ids
        p_ids = np.random.choice(min(p1.shape[0], p2.shape[0]), size=k, replace=False)

        # write chosen points to sample arrays
        for i, ids in enumerate(p_ids):
            sample1[i] = p1[ids]
            sample2[i] = p2[ids]

        return sample1, sample2

    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormalized cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
        # calculate mean and half absolute value per component
        t = np.mean(points, axis=0)
        s = 1/2 * np.max(np.abs(points), axis=0)

        # create ps and T matrices
        ps = np.zeros((points.shape[0], 3))
        T = np.zeros((3, 3))

        # compute normalized points in homogeneous coordinates
        for l in range(points.shape[0]):
            ps[l, 0] = (points[l, 0] - t[0])/s[0]
            ps[l, 1] = (points[l, 1] - t[1])/s[1]
            ps[l, 2] = 1

        # compute T matrix
        T[0, 0] = 1/s[0]
        T[1, 1] = 1/s[1]
        T[2, 2] = 1
        T[0, 2] = -1 * t[0] / s[0]
        T[1, 2] = -1 * t[1] / s[1]

        return ps, T

    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices shoul be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.

        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        # build matrix  out of l point correspondences
        A = np.zeros((2*p1.shape[0], 9))
        for l in range(p1.shape[0]):
            # fill matrix with values according to lecture 8 slide 14
            A[2*l, 3] = p1[l, 0]
            A[2*l, 4] = p1[l, 1]
            A[2*l, 5] = 1
            A[2*l, 6] = -1 * p1[l, 0] * p2[l, 1]
            A[2*l, 7] = -1 * p1[l, 1] * p2[l, 1]
            A[2*l, 8] = -1 * p2[l, 1]
            A[2*l+1, 0] = -1 * p1[l, 0]
            A[2*l+1, 1] = -1 * p1[l, 1]
            A[2*l+1, 2] = -1
            A[2*l+1, 6] = p1[l, 0] * p2[l, 0]
            A[2*l+1, 7] = p1[l, 1] * p2[l, 0]
            A[2*l+1, 8] = p2[l, 0]

        # perform SVD
        u, s, vh = np.linalg.svd(A)
        # solution is the last right singular vector
        h = vh[-1]
        # reshape into homography matrix (conditioned)
        HC = h.reshape((3, 3), order='C')
        # compute homography matrix (unconditioned)
        H = np.linalg.inv(T2) @ HC @ T1

        # normalize so that matrix element (2, 2) is equal to one
        H = H / H[2, 2]
        HC = HC / HC[2, 2]

        return H, HC

    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

        # initialize points array
        points = np.zeros((p.shape[0], 2))

        # iterate through interest point array p
        for l in range(p.shape[0]):
            # use homogeneous coordinates
            p_hom = np.array([[p[l, 0]], [p[l, 1]], [1]])
            # compute transformation using the homography matrix
            p_transformed = H @ p_hom
            # store x and y values for solution
            # normalize by z value to obtain cartesian coordinates
            points[l, 0] = p_transformed[0]/p_transformed[2]
            points[l, 1] = p_transformed[1]/p_transformed[2]

        return points

    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        # initialize dist array
        dist = np.zeros((p1.shape[0],))

        # transform points
        p1_transformed = self.transform_pts(p1, H)
        p2_transformed = self.transform_pts(p2, np.linalg.inv(H))

        # compute distance for every interest point pair using the precomputed transformed points
        for l in range(p1.shape[0]):
            # compute distance according to formula in the exercise sheet
            dist[l] = (np.linalg.norm(p1_transformed[l] - p2[l]))**2 + (np.linalg.norm(p1[l] - p2_transformed[l]))**2

        return dist

    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 

        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """

        # initialize N and inliers (inliears has to be build dynamically, because N is not known)
        N = 0
        inliers = None

        # iterate through all keypoint pairs l
        for l in range(pairs.shape[0]):
            # only considered inlier if distance is smaller than a threshold
            if dist[l] < threshold:
                # increase N and add found keypoint pair to set of inliers
                N += 1
                if inliers is None:
                    inliers = pairs[l].reshape(1, 4)
                else:
                    inliers = np.append(inliers, pairs[l].reshape(1, 4), axis=0)

        return N, inliers

    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        # computed following lecture 8 slide 23
        # use ceiling function to take next integer value above computed threshold
        return int(np.ceil(np.log(1-z)/np.log(1-(p**k))))

    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.

        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """

        # initialize H, max_inliers and inliers
        H = None
        max_inliers = 0
        inliers = None

        # following cookbook recipe in lecture 8 slide 25

        # condition coordinates
        p1, T1 = self.condition_points(pairs[:, :2])
        p2, T2 = self.condition_points(pairs[:, 2:4])

        # run ransac algorithm for n_iters
        for n in range(n_iters):
            # random sample of k points
            # use cartesian coordinates (we know that third entry is 1 so can just ignore -> already normalized)
            sample1, sample2 = self.pick_samples(p1[:, :2], p2[:, :2], k)

            # compute homography
            temp_H, temp_HC = self.compute_homography(sample1, sample2, T1, T2)

            # compute distances
            #dist = self.compute_homography_distance(temp_HC, p1[:, :2], p2[:, :2])
            dist = self.compute_homography_distance(temp_H, pairs[:, :2], pairs[:, 2:4])
            # evaluate homography
            tmp_N, tmp_inliers = self.find_inliers(pairs, dist, threshold)

            # check if computed homography has max inliers and save if true
            if tmp_N > max_inliers:
                max_inliers = tmp_N
                inliers = tmp_inliers
                H = temp_H

        return H, max_inliers, inliers

    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """

        # condition coordinates
        p1, T1 = self.condition_points(inliers[:, :2])
        p2, T2 = self.condition_points(inliers[:, 2:4])

        # recompute homography
        H, HC = self.compute_homography(p1, p2, T1, T2)

        return H

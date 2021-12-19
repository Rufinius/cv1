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

        d = np.zeros((features2.shape[1], features1.shape[1]))
        for i in range(features2.shape[1]):
            for j in range(features1.shape[1]):
                d[i,j] = np.linalg.norm(features1.T[j]-features2.T[i])**2

        return d

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

        pairs = np.zeros((p1.shape[0], 4))
        min_d = float('inf')
        loc = 0
        for i in range(p1.shape[0]):
            for j in range(p2.shape[0]):
                if distances[j,i] < min_d:
                    min_d = distances[j,i]
                    loc = j
            pairs[i,0] = p1[i,0]
            pairs[i,1] = p1[i,1]
            pairs[i,2] = p2[loc, 0]
            pairs[i,3] = p2[loc, 1]
            loc = 0
            min_d = float('inf')

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
        
        sample1 = []
        sample2 = []

        for i in range(k):
            if p1.shape[0] > p2.shape[0]:
                loc = np.random.choice(p2.shape[0])
            else:
                loc = np.random.choice(p1.shape[0])
            sample1.append(p1[loc, :])
            sample2.append(p2[loc, :])

        return np.array(sample1), np.array(sample2)


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.

        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """

        test = np.apply_along_axis(np.linalg.norm, 1, points)
        test1 = max(test)

        s = 0.5 * max(np.apply_along_axis(np.linalg.norm, 1, points))
        t = np.mean(points, axis=0)
        T = np.array([[1/s, 0, -t[0]/s], [0, 1/s, -t[1]/s], [0, 0, 1]])
        u = np.ones((points.shape[0], 3))
        u[:, 0] = points[:, 0]
        u[:, 1] = points[:, 1]

        for i in range(points.shape[0]):
            u[i] = T @ u[i]

        return u, T


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

        H = np.zeros((p1.shape[0]*2, 9))
        count = 0
        for i in range(0, p1.shape[0]*2, 2):
            H[i, 0] = 0
            H[i, 1] = 0
            H[i, 2] = 0
            H[i, 3] = p1[count, 0]
            H[i, 4] = p1[count, 1]
            H[i, 5] = 1
            H[i, 6] = -p1[count, 0] * p2[count, 1]
            H[i, 7] = -p1[count, 1] * p2[count, 1]
            H[i, 8] = -p2[count, 1]
            H[i+1, 0] = -p1[count, 0]
            H[i+1, 1] = -p1[count, 1]
            H[i+1, 2] = -1
            H[i+1, 3] = 0
            H[i+1, 4] = 0
            H[i+1, 5] = 0
            H[i+1, 6] = p1[count, 0] * p2[count, 0]
            H[i+1, 7] = p1[count, 1] * p2[count, 0]
            H[i+1, 8] = p2[count, 0]
            count += 1

        u, s, v = np.linalg.svd(H)
        h = v[-1]
        h_uncond = np.linalg.inv(T2) @ h.reshape((3, 3)) @ T1
        h = h / h[-1]
        h_uncond = h_uncond / h_uncond[2, 2]

        return h_uncond, h.reshape((3, 3))


    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  

        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """

        p_hom = np.ones((p.shape[0], 3))
        p_hom[:, 0] = p[:, 0]
        p_hom[:, 1] = p[:, 1]

        points = np.zeros((p.shape[0], 3))

        for i in range(p.shape[0]):
            points[i] = H @ p_hom[i]
            points[i] = points[i]/points[i, 2]

        return np.column_stack((points[:, 0], points[:, 1]))




    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.

        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """

        p1_hom = np.ones((p1.shape[0], 3))
        p1_hom[:, 0] = p1[:, 0]
        p1_hom[:, 1] = p1[:, 1]
        p2_hom = np.ones((p2.shape[0], 3))
        p2_hom[:, 0] = p2[:, 0]
        p2_hom[:, 1] = p2[:, 1]

        distances = np.zeros((p1.shape[0]))

        for i in range(p1.shape[0]):
            distances[i] = np.linalg.norm(H @ p1_hom[i] - p2_hom[i])**2 + \
                   np.linalg.norm(p1_hom[i] - np.linalg.inv(H) @ p2_hom[i])**2

        return distances


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

        count = 0
        inliers = []

        for i in range(pairs.shape[0]):
            if dist[i] < threshold:
                count += 1
                inliers.append(pairs[i])

        return count, inliers


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.

        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """

        return np.round(np.log(1-z)/np.log(1-p**k))



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

        p1 = pairs[:,0:2:]
        p2 = pairs[:, 2:4:]
        p1_cond, T1 = self.condition_points(p1)
        p2_cond, T2 = self.condition_points(p2)

        best_H = 0
        max_count = 0
        max_inliers = 0

        for i in range(int(n_iters)):
            samp1, samp2 = self.pick_samples(p1_cond[:, 0:2:], p2_cond[:, 0:2:], k)
            H, HC = self.compute_homography(samp1, samp2, T1, T2)
            p1_trans = self.transform_pts(samp1, np.linalg.inv(HC))
            #p2_trans = self.transform_pts(samp2, HC)
            distance = self.compute_homography_distance(HC, p1_trans, samp2)
            count, inliers = self.find_inliers(np.column_stack([p1_trans, samp2]), distance, threshold)
            if count > max_count:
                best_H = H
                max_count = count
                max_inliers = inliers
            print(i)

        return H, max_count, max_inliers





    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.

        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
    #
    # You code here
    #
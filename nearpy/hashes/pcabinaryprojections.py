# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pickle
import numpy as np
from itertools import izip

from nearpy.hashes.lshash import LSHash
from nearpy.utils import perform_online_pca


class PCABinaryProjections(LSHash):
    """
    Projects a vector on n first principal components and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each principal component hyperplane and
    generates a binary hash value in string form, which is being
    used as a bucket key for storage.
    """

    def __init__(self, name, dimension, trainset, nbits, pkl=None):
        super(PCABinaryProjections, self).__init__(name)
        self.dimension = dimension
        self.nbits = nbits

        if pkl is not None:
            self.mean, (self.eigenvalues, self.eigenvectors) = pickle.load(open(pkl))
        else:
            self.mean, (self.eigenvalues, self.eigenvectors) = perform_online_pca(trainset(), dimension)
            pickle.dump((self.mean, (self.eigenvalues, self.eigenvectors)), open("pca.pkl", 'w'))

        #variance_explanation = np.cumsum(eigenvalues/eigenvalues.sum())
        #self.projection_count = np.sum(variance_explanation <= max_explanation)

        self.npca = min(self.nbits, self.dimension)  # Number of principal components to keep.
        self.normals = self.eigenvectors.T[:self.npca]

    def hash_vector(self, V, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        projections = V.reshape((-1, self.dimension)) - self.mean
        projections = np.dot(projections, self.normals.T)

        # Return binary key as a string
        projections = (projections > 0.0).astype(np.int8).astype('S1')
        return [''.join(projection) for projection in projections]

    def hash_vector_with_pos(self, V, positions, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        projections = V.reshape((-1, self.dimension)) - self.mean
        projections = np.dot(projections, self.normals.T)

        # Return binary key as a string
        projections = (projections > 0.0).astype(np.int8).astype('S1')
        return ["_".join(map(str, pos)) + "_" + ''.join(projection) for projection, pos in izip(projections, positions)]

    def __str__(self):
        text = ""
        text += self.name + ": " + str(self.nbits)
        return text

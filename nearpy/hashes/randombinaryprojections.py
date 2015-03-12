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

import numpy as np
from itertools import izip

from nearpy.hashes.lshash import LSHash


class RandomBinaryProjections(LSHash):
    """
    Projects a vector on n random hyperplane normals and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each hyperplane and generates a binary
    hash value in string form, which is being used as a bucket key
    for storage.
    """

    def __init__(self, name, projection_count, dimension, rand_seed=None):
        """
        Creates projection_count random vectors, that are used for projections
        thus working as normals of random hyperplanes. Each random vector /
        hyperplane will result in one bit of hash.

        So if you for example decide to use projection_count=10, the bucket
        keys will have 10 digits and will look like '1010110011'.
        """
        super(RandomBinaryProjections, self).__init__(name)
        self.projection_count = projection_count
        self.dimension = dimension
        self.rand = np.random.RandomState(rand_seed)
        self.normals = self.rand.randn(self.projection_count, self.dimension).astype(np.float32)

    def hash_vector(self, V, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        projections = np.dot(V.reshape((-1, self.dimension)), self.normals.T)

        # Return binary key as a string
        projections = (projections > 0.0).astype(np.int8).astype('S1')
        return [''.join(projection) for projection in projections]

    def hash_vector_with_pos(self, V, positions, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        projections = np.dot(V.reshape((-1, self.dimension)), self.normals.T)

        # Return binary key as a string
        projections = (projections > 0.0).astype(np.int8).astype('S1')
        return ["_".join(map(str, pos)) + "_" + ''.join(projection) for projection, pos in izip(projections, positions)]

    def __str__(self):
        text = ""
        text += self.name + ": " + str(self.projection_count)
        return text

# -*- coding: utf-8 -*-

import numpy as np

import nearpy.utils.utils as utils
from nearpy.hashes.hashing import Hashing


class LocalitySensitiveHashing(Hashing):
    """
    Projects a vector on n random hyperplane normals and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each hyperplane and generates a binary
    hash value in string form, which is being used as a bucket key
    for storage.
    """

    def __init__(self, name, dimension, nbits, rand_seed=None):
        super(LocalitySensitiveHashing, self).__init__(name, nbits)
        self.dimension = dimension
        self.rand = np.random.RandomState(rand_seed)
        self.normals = self.rand.randn(self.dimension, self.nbits).astype(np.float32)

    def hash_vector(self, V):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        with utils.Timer("    Projecting"):
            projections = np.dot(V, self.normals)

        # Convert bitcode to uint
        with utils.Timer("    Thresholding"):
            projections = np.dot(projections > 0, self.bits_to_int)

        # Return the hashcode view as a string
        with utils.Timer("    Stringifying"):
            projections = projections.view("|S8")

        return projections

    # def hash_vector_with_pos(self, V, positions, querying=False):
    #     """
    #     Hashes the vector and returns the binary bucket key as string.
    #     """
    #     projections = np.dot(V.reshape((-1, self.dimension)), self.normals.T)

    #     # Return binary key as a string
    #     projections = (projections > 0.0).astype(np.int8).astype('S1')
    #     return ["_".join(map(str, pos)) + "_" + ''.join(projection) for projection, pos in izip(projections, positions)]

    def __str__(self):
        text = ""
        text += self.name + ": " + str(self.nbits)
        return text

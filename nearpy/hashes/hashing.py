# -*- coding: utf-8 -*-

import numpy as np


class Hashing(object):
    """ Interface for hashing functions. """

    def __init__(self, name, nbits):
        """
        The hash name is used in storage to store buckets of
        different hashes without collision.
        """
        self.name = name
        self.nbits = nbits

        # It's more efficient to store uint than bitcodes represented as string.
        self.bits_to_int = np.array([np.uint(2**i) for i in range(self.nbits)])

    def hash_vector(self, V):
        """
        Hashes the vector and returns a list of bucket keys, that match the
        vector. Depending on the hash implementation this list can contain
        one or many bucket keys. Querying is True if this is used for
        retrieval and not indexing.
        """
        raise NotImplementedError

    def __getstate__(self):
        state = {}
        state.update(self.__dict__)
        state["hashing_version"] = 1
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

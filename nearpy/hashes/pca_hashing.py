# -*- coding: utf-8 -*-

import pickle
import numpy as np
from itertools import izip

import nearpy.utils.utils as utils

from nearpy.hashes.hashing import Hashing
from nearpy.utils import perform_online_pca


class PCAHashing(Hashing):
    """
    Projects a vector on n first principal components and assigns
    a binary value to each projection depending on the sign. This
    divides the data set by each principal component hyperplane and
    generates a binary hash value in string form, which is being
    used as a bucket key for storage.
    """

    def __init__(self, name, dimension, trainset, nbits, pca_pkl=None):
        super(PCAHashing, self).__init__(name, nbits)
        self.dimension = dimension
        self.npca = min(self.nbits, self.dimension)  # Number of principal components to keep.

        if pca_pkl is not None:
            self.mean, (self.eigenvalues, self.eigenvectors) = pickle.load(open(pca_pkl))
        else:
            self.mean, (self.eigenvalues, self.eigenvectors) = perform_online_pca(trainset(), dimension)
            pickle.dump((self.mean, (self.eigenvalues, self.eigenvectors)), open("pca.pkl", 'w'))

        #variance_explanation = np.cumsum(eigenvalues/eigenvalues.sum())
        #self.projection_count = np.sum(variance_explanation <= max_explanation)

        try:
            # GPU version is ~100x faster (requires Theano)
            self.hash_func = self._build_hash_function()
        except:
            self.hash_func = None

    def _build_hash_function(self):
        import theano
        import theano.tensor as T

        # Make sure everything is in float32 (i.e. GPU compatible)
        eigenvectors = self.eigenvectors.astype(np.float32)
        mean = self.mean.astype(np.float32)
        V = T.matrix()

        # Keep the first `self.npca` principal components.
        proj = eigenvectors[:, :self.npca]
        projections = T.dot(V-mean, proj)
        f = theano.function([V], projections)
        return f

    def hash_vector(self, V):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
        projections = V.reshape((-1, self.dimension)) - self.mean
        projections = np.dot(projections, self.normals.T)

        # Return binary key as a string
        projections = (projections > 0.0).astype(np.int8).astype('S1')
        return [''.join(projection) for projection in projections]

        with utils.Timer("    Projecting"):
            if self.hash_func is not None:
                # Use GPU (~100x faster)
                projections = []
                for chunk in utils.chunk(V, 200000):
                    projections.append(self.hash_func(chunk))

                projections = np.concatenate(projections, axis=0)
            else:
                # Keep the first `self.npca` principal components.
                proj = self.eigenvectors[:, :self.npca]
                projections = np.dot(V-self.mean, proj)

        # Convert bitcode to uint
        with utils.Timer("    Thresholding"):
            projections = np.dot(projections > 0, self.bits_to_int)

        # Return the hashcode view as a string
        with utils.Timer("    Stringifying"):
            projections = projections.view("|S8")

        return projections

    def __str__(self):
        text = ""
        text += self.name + ": " + str(self.nbits)
        return text

    def __getstate__(self):
        state = super(PCAHashing, self).__getstate__()
        state["PCAHashing_version"] = 1
        return state

    def __setstate__(self, state):
        super(PCAHashing, self).__setstate__(state)

        if "PCAHashing_version" not in state:
            if 'projection_count' in state:
                self.nbits = state['projection_count']
        else:
            if state["PCAHashing_version"] >= 1:
                try:
                    # GPU version is ~100x faster (requires Theano)
                    self.hash_func = self._build_hash_function()
                except:
                    self.hash_func = None

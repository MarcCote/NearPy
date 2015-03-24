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
import nearpy.utils.utils as utils

from nearpy.hashes import PCABinaryProjections


class SpectralHashing(PCABinaryProjections):
    """
    This is essentially a Python translation of the MATLAB code of [Weiss2008b]_.

    References
    ----------
    .. [Weiss2008a] Weiss, Y., Torralba, A., & Fergus, R. (2008). Spectral Hashing. NIPS, (1), 1–8.
    .. [Weiss2008b] http://www.cs.huji.ac.il/~yweiss/SpectralHashing/
    """
    def __init__(self, name, bounds=None, *args, **kwargs):
        super(SpectralHashing, self).__init__(name, *args, **kwargs)

        self.bits_to_int = np.array([np.uint(2**i) for i in range(self.nbits)])

        # Compute the bounding box's bounds of the data in PCA space
        if bounds is None:
            bounds = [np.inf * np.ones(self.npca, dtype="float32"),
                      -np.inf * np.ones(self.npca, dtype="float32")]

            # Keep the first `self.npca` principal components.
            proj = self.eigenvectors[:, :self.npca]

            # Find optimal bounds by going through a trainset.
            for V in kwargs['trainset']():
                projV = np.dot(V, proj)  # According to Weiss, no need to remove the mean.
                bounds[0] = np.minimum(bounds[0], np.min(projV, axis=0, keepdims=True))
                bounds[1] = np.maximum(bounds[1], np.max(projV, axis=0, keepdims=True))

            pickle.dump(bounds, open('bounds.pkl', 'w'))
        else:
            bounds = pickle.load(open(bounds))

        eps = 1e-8
        self.bounds = (bounds[0]-eps, bounds[1]+eps)
        #self.bits_to_int = np.array([np.uint(2**i) for i in range(self.nbits)])

        R = bounds[1] - bounds[0]

        maxMode = np.ceil((self.nbits+1)*R/max(R))
        nModes = np.sum(maxMode)-len(maxMode)+1
        modes = np.ones((nModes, self.npca))

        m = 0
        for i in range(self.npca):
            modes[(m+1):(m+maxMode[i]), i] = np.arange(2, maxMode[i]+1)
            m += maxMode[i]-1

        modes = modes - 1
        omega0 = np.pi/R
        omegas = modes*np.tile(omega0, (nModes, 1))
        eigVal = -np.sum(omegas**2, 1)
        ii = np.argsort(-eigVal)
        modes = modes[ii[1:(self.nbits+1)], :]

        self.modes = modes

        def build_hash_function():
            import theano
            import theano.tensor as T

            # Make sure everything is in float32 (i.e. GPU compatible)
            modes = self.modes.astype(np.float32)
            eigenvectors = self.eigenvectors.astype(np.float32)
            bounds = (self.bounds[0].astype(np.float32), self.bounds[1].astype(np.float32))

            V = T.matrix()

            # Keep the first `self.npca` principal components.
            proj = eigenvectors[:, :self.npca]
            projV = T.dot(V, proj)[:, None, :]  # According to Weiss, no need to remove the mean.

            pi = np.float32(np.pi)
            half_pi = np.float32(np.pi/2.)
            omega0 = pi / (bounds[1] - bounds[0])
            omegas = omega0 * modes
            projections = T.prod(T.sin(omegas*(projV-bounds[0])+half_pi), axis=2)
            f = theano.function([V], projections)
            return f

        try:
            self.hash_func = build_hash_function()
        except:
            self.hash_func = None

    def hash_vector(self, V, querying=False):
        """
        Hashes the vector and returns the binary bucket key as string.
        """

        #V = V.reshape((-1, self.dimension))
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
                projV = np.dot(V, proj)[:, None, :]  # According to Weiss, no need to remove the mean.

                pi = np.float32(np.pi)
                half_pi = np.float32(np.pi/2.)
                omega0 = pi / (self.bounds[1] - self.bounds[0])
                omegas = omega0 * self.modes
                projections = np.prod(np.sin(omegas*(projV-self.bounds[0])+half_pi), axis=2)

        self.bits_to_int = np.array([np.uint(2**i) for i in range(self.nbits)])
        # Return binary key as a string
        with utils.Timer("    Thresholding"):
            #projections = (projections > 0.0).astype(np.int8).astype('S1')
            projections = np.dot(projections > 0, self.bits_to_int)

        with utils.Timer("    Stringifying"):
            #projections = [''.join(projection) for projection in projections]
            #projections = map(str, projections)
            #projections = list(utils.chunk(projections.tostring(), projections.itemsize))
            projections = projections.view("|S8")

        return projections

    def __str__(self):
        text = ""
        text += self.name + ": " + str(self.nbits)
        return text

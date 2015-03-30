# -*- coding: utf-8 -*-

import pickle
import numpy as np
import nearpy.utils.utils as utils

from nearpy.hashes import PCAHashing


class SpectralHashing(PCAHashing):
    """
    This is essentially a Python translation of the MATLAB code of [Weiss2008b]_.

    References
    ----------
    .. [Weiss2008a] Weiss, Y., Torralba, A., & Fergus, R. (2008). Spectral Hashing. NIPS, (1), 1–8.
    .. [Weiss2008b] http://www.cs.huji.ac.il/~yweiss/SpectralHashing/
    """
    def __init__(self, name, bounds_pkl=None, *args, **kwargs):
        super(SpectralHashing, self).__init__(name, *args, **kwargs)

        # Compute the bounding box's bounds of the data in PCA space
        if bounds_pkl is None:
            bounds = [np.inf * np.ones(self.npca, dtype="float32"),
                      -np.inf * np.ones(self.npca, dtype="float32")]

            # Keep the first `self.npca` principal components.
            proj = self.eigenvectors[:, :self.npca]

            # Find optimal bounds by going through a trainset.
            for V in kwargs['trainset']():
                projV = np.dot(V, proj)  # According to Weiss, no need to remove the mean.
                bounds[0] = np.minimum(bounds[0], np.min(projV, axis=0))
                bounds[1] = np.maximum(bounds[1], np.max(projV, axis=0))

            pickle.dump(bounds, open('bounds.pkl', 'w'))
        else:
            bounds = pickle.load(open(bounds_pkl))

        eps = 1e-8
        self.bounds = (bounds[0]-eps, bounds[1]+eps)

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

        try:
            # GPU version is ~100x faster (requires Theano)
            self.hash_func = self.build_hash_function()
        except:
            self.hash_func = None

    def _build_hash_function(self):
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

    def hash_vector(self, V):
        """
        Hashes the vector and returns the binary bucket key as string.
        """
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
        state = super(SpectralHashing, self).__getstate__()
        state["SpectralHashing_version"] = 1
        return state

    def __setstate__(self, state):
        super(SpectralHashing, self).__setstate__(state)

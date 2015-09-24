import numpy as np

from nearpy.distances.distance import Distance


class CorrelationDistance(Distance):
    """ Correlation distance """

    def __call__(self, query, patches):
        axis = tuple(range(1, patches.ndim))
        mean1 = np.mean(query)
        mean2 = np.mean(patches, axis=axis, keepdims=True)
        std1 = np.std(query)
        std2 = np.std(patches, axis=axis, keepdims=True)

        return np.mean( ((query - mean1) / std1) * ((patches - mean2) / std2), axis=axis)

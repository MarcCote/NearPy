# -*- coding: utf-8 -*-
import numpy as np
from nearpy.filters.vectorfilter import VectorFilter


class SortedFilter(VectorFilter):
    """
    Sorts vectors with respect to distance.
    """

    def __call__(self, distances):
        return np.argsort(distances)

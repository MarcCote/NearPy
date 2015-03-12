
import numpy as np
from nearpy.filters.vectorfilter import VectorFilter


class PositionFilter(VectorFilter):
    """
    Sorts vectors with respect to distance and returns the N nearest.
    """

    def __init__(self, radius, attribute):
        """
        Keeps the count threshold.
        """
        self.radius = radius
        self.attribute = attribute

    def __call__(self, neighbors):
        """
        Returns subset of specified input list.
        """
        neighbors[attribute.name]
        return np.argsort(distances)[:self.N]

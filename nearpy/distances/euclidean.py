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

# import theano
# import theano.tensor as T
import numpy as np

from nearpy.distances.distance import Distance


class EuclideanDistance(Distance):
    """ Euclidean distance """

    # def __init__(self, *args, **kwargs):
    #     super(EuclideanDistance, self).__init__(*args, **kwargs)
    #     patches = T.matrix()
    #     query = T.vector()
    #     distances = T.sqrt(T.mean((patches - query) ** 2, axis=1))
    #     #self.dist = theano.function([patches, query], distances)

    #     self.dist = theano.function([patches, query], T.argsort(distances)[:100])

    #     theano.printing.pydotprint(self.dist)

    def __call__(self, query, patches):
        return np.sqrt(np.mean((patches - query) ** 2, axis=tuple(range(1, patches.ndim))))
        #return np.sqrt(np.sum((patches - query) ** 2, axis=tuple(range(1, patches.ndim))))
        #return self.dist(patches.reshape((-1, 9)), query.flatten())

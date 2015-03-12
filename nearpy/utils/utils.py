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

import sys
import json
import numpy
import numpy as np
from itertools import islice


def numpy_array_from_list_or_numpy_array(vectors):
    """
    Returns numpy array representation of argument.

    Argument maybe numpy array (input is returned)
    or a list of numpy vectors.
    """
    # If vectors is not a numpy matrix, create one
    if not isinstance(vectors, numpy.ndarray):
        V = numpy.zeros((vectors[0].shape[0], len(vectors)))
        for index in range(len(vectors)):
            vector = vectors[index]
            V[:, index] = vector
        return V

    return vectors


def perform_pca(A):
    """
    Computes eigenvalues and eigenvectors of covariance matrix of A.
    The rows of a correspond to observations, the columns to variables.
    """
    # First subtract the mean
    M = (A-numpy.mean(A, axis=0)).T
    # Get eigenvectors and values of covariance matrix
    return numpy.linalg.eig(numpy.cov(M))


def perform_online_pca(data, dimension=None):
    if dimension is None:
        dimension = data[0].shape

    total = 0
    mean = np.zeros(dimension, dtype=np.float64)
    comoment = np.zeros((dimension, dimension), dtype=np.float64)

    for element in data:
        total += len(element)
        last_mean = mean
        mean = mean + np.sum(element - mean, axis=0, dtype=np.float64)/total
        comoment += np.dot((element-mean).T, (element-last_mean))

    cov = comoment / (total-1)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    ordered_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[ordered_idx]
    eigenvectors = eigenvectors[:, ordered_idx]

    return mean, (eigenvalues, eigenvectors)


PY2 = sys.version_info[0] == 2
if PY2:
    bytes_type = str
else:
    bytes_type = bytes


def want_string(arg, encoding='utf-8'):
    if isinstance(arg, bytes_type):
        rv = arg.decode(encoding)
    else:
        rv = arg
    return rv


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in xrange(0, len(sequence), n):
        yield sequence[i:i + n]


def ichunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    sequence = iter(sequence)
    chunk = list(islice(sequence, n))
    while len(chunk) > 0:
        yield chunk
        chunk = list(islice(sequence, n))


def load_dict_from_json(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


def save_dict_to_json(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))

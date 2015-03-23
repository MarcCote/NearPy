# -*- coding: utf-8 -*-
from __future__ import print_function

import json
import sys
import numpy
import numpy as np
from time import time
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
    try:
        with open(path, "r") as json_file:
            return json.loads(json_file.read())
    except:
        print("infos.json is corrupted")

    return {}


def save_dict_to_json(path, dictionary):
    data = json.dumps(dictionary, indent=4, separators=(',', ': '))
    with open(path, "w") as json_file:
        json_file.write(data)


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))

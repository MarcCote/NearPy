import numpy as np

from nearpy.utils import chunk


class Data(object):
    def __init__(self, name):
        self.name = name

    def dumps(self, data):
        return data

    def loads(self, txt):
        return eval(txt)


class NumpyData(Data):
    def __init__(self, name, dtype, shape):
        super(NumpyData, self).__init__(name)
        self.dtype = dtype
        self.shape = shape

    def dumps(self, data):
        return chunk(data.tostring(), n=int(np.prod(self.shape)*self.dtype.itemsize))

    def loads(self, txt):
        shape = (-1,) + self.shape
        return np.frombuffer(txt, self.dtype).reshape(shape)

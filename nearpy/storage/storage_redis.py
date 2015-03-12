# -*- coding: utf-8 -*-

# Copyright (c) 2013 Ole Krause-Sparmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
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

try:
    import ujson as json
except ImportError:
    import json

import types
import numpy as np
from redis import Redis

from itertools import izip
from nearpy.storage.storage import Storage
from nearpy.utils import chunk

# def serialize(x):
#     if type(x) is np.ndarray:
#         return 'nd' + x.dtype.str + '{:02}'.format(x.ndim) + str(x.shape) + x.tostring()

# def deserialize(s):
#     if s[:2] == 'nd':
#         return s


class RedisStorage(Storage):
    """ Storage using redis. """

    def __init__(self, redis_object=None, host='localhost', port=6379, db=0, keyprefix="nearpy_"):
        """ Uses specified redis object for storage. """
        if redis_object is None:
            redis_object = Redis(host=host, port=port, db=db)

        self.redis_object = redis_object
        self.keyprefix = keyprefix

    # def __setitem__(self, bucket_key, bucket_value):
    #     return self.store_buckets(**{bucket_key: bucket_value})

    def append_to_bucket(self, bucket_key, bucket_value):
        """
        Parameters
        ----------
        bucket_key: str
            key of the bucket
        bucket_value: JSON serializable object
            information to append to the bucket refered by `bucket_key`

        Return
        ------
        count: int
            number of elements added
        """
        return self.append_to_buckets_from_iter([(bucket_key, bucket_value)])

    # def store(self, buckets):
    #     count = 0
    #     pipeline = self.redis_object.pipeline(transaction=False)
    #     for bucket_key, bucket_data in buckets.items():
    #         key = self.key_prefix + bucket_key + '_'
    #         for name, data in bucket_data.items():
    #             pipeline.rpush(key + name, *map(json.dumps, data))

    #         count += len(data)

    #     pipeline.execute()
    #     return count

    # def store(self, buckets, attribute):
    #     count = 0
    #     pipeline = self.redis_object.pipeline(transaction=False)
    #     for bucket_key, bucket_data in buckets.items():
    #         key = self.key_prefix + bucket_key + '_' + attribute
    #         pipeline.rpush(key, *map(json.dumps, bucket_data))
    #         count += len(bucket_data)

    #     pipeline.execute()
    #     return count

    # def retrieve(self, bucket_keys, attribute):
    #     pipeline = self.redis_object.pipeline(transaction=False)
    #     for bucket_key in bucket_keys:
    #         key = self.key_prefix + bucket_key + '_' + attribute
    #         pipeline.lrange(key, 0, -1)

    #     results = pipeline.execute()
    #     return map(lambda items: map(json.loads, items), results)

    def store(self, bucketkeys, prefix="", **bucketvalues):
        #chunk_size = 100000
        keys = [self.keyprefix + prefix + bucketkey for bucketkey in bucketkeys]
        pipeline = self.redis_object.pipeline(transaction=False)
        for attribute, values in bucketvalues.items():
            for key, value in izip(keys, values):
        #    for chunk_key, chunk_value in izip(chunk(keys, n=chunk_size), chunk(values, n=chunk_size)):
        #        for key, value in izip(chunk_key, chunk_value):
                pipeline.rpush(key + "_" + attribute, value)

        pipeline.execute()

        return len(bucketkeys)

    def retrieve(self, bucketkeys, attribute, prefix=""):
        keys = [self.keyprefix + prefix + bucketkey for bucketkey in bucketkeys]
        pipeline = self.redis_object.pipeline(transaction=False)
        for key in keys:
            pipeline.lrange(key + '_' + attribute, 0, -1)

        results = pipeline.execute()
        #return map(lambda items: map(json.loads, items), results)
        return results

    def set_metadata(self, key, metadata):
        for field, value in metadata.items():
            self.redis_object.hsetnx(key, field, value)

    def get_metadata(self, key):
        return self.redis_object.hgetall(key)

    def add_attribute(self, key, attribute):
        return self.redis_object.sadd(key + "_attributes", attribute)

    def get_attributes(self, key):
        return self.redis_object.smembers(key + "_attributes")

    def append_to_buckets_from_iter(self, bucket_keyvalues):
        """
        Parameters
        ----------
        bucket_keyvalues: iterable of tuples
            key-value pairs (str, JSON serializable object)

        Return
        ------
        count: int
            number of elements added
        """
        pipeline = self.redis_object.pipeline(transaction=False)
        #import time
        #start = time.time()
        for count, (bucket_key, bucket_value) in enumerate(bucket_keyvalues, start=1):
            key = 'nearpy_' + bucket_key
            pipeline.rpush(key, json.dumps(bucket_value))
            #pipeline.rpush(key, bucket_value)

        #print "#{}, {:.2f} sec.".format(count, time.time()-start)

        pipeline.execute()
        return count

    def retrieve2(self, bucket_keys):
        """
        Parameters
        ----------
        bucket_keys: iterable of string
            keys from which to retrieve values

        Return
        ------
        values: list of JSON serializable objects
            values retrieved
        """
        single_key = False
        if not isinstance(bucket_keys, types.ListType) and not isinstance(bucket_keys, types.GeneratorType):
            single_key = True
            bucket_keys = [bucket_keys]

        pipeline = self.redis_object.pipeline(transaction=False)
        for bucket_key in bucket_keys:
            key = 'nearpy_' + bucket_key
            pipeline.lrange(key, 0, -1)

        values = pipeline.execute()
        if single_key:
            return map(json.loads, values[0])
            #return values[0]

        return [map(json.loads, value) for value in values]
        #return values

    def count(self, bucket_keys=[], prefix=None):
        """
        Parameters
        ----------
        bucket_keys: iterable of string
            keys from which to retrieve values
        prefix: string
            if set, report counts of every buckets having this prefix

        Return
        ------
        counts: list of int
            size of each given bucket
        """

        single_key = False
        pipeline = self.redis_object.pipeline(transaction=False)
        if prefix is not None:
            keys = self.redis_object.keys(pattern='nearpy_' + prefix + "*")
            for key in keys:
                pipeline.llen(key)

        else:
            if not isinstance(bucket_keys, types.ListType) and not isinstance(bucket_keys, types.GeneratorType):
                single_key = True
                bucket_keys = [bucket_keys]

            for bucket_key in bucket_keys:
                key = 'nearpy_' + bucket_key
                pipeline.llen(key)

        counts = pipeline.execute()

        if single_key:
            return counts[0]

        return counts

    def buckets_count(self, prefix):
        """
        Parameters
        ----------
        prefix: string
            report buckets count having this prefix

        Return
        ------
        count: int
            number of buckets
        """
        return len(self.redis_object.keys(pattern='nearpy_' + prefix + "*"))

    def clear(self, bucket_keys=[], prefix=None):
        """
        Parameters
        ----------
        bucket_keys: iterable of string
            keys from which to retrieve values
        prefix: string
            if set, clear every buckets having this prefix

        Return
        ------
        count: int
            number of buckets cleared
        """

        pipeline = self.redis_object.pipeline(transaction=False)
        if prefix is not None:
            keys = self.redis_object.keys(pattern='nearpy_' + prefix + "*")
            for key in keys:
                pipeline.delete(key)

        else:
            if not isinstance(bucket_keys, types.ListType) and not isinstance(bucket_keys, types.GeneratorType):
                bucket_keys = [bucket_keys]

            for bucket_key in bucket_keys:
                key = 'nearpy_' + bucket_key
                pipeline.delete(key)

        counts = pipeline.execute()
        return sum(counts)

    # def save(self):
    #     try:
    #         self.redis_object.save()
    #     except ResponseError:
    #         pass  # Saveing already in progress!

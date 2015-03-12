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

import json
import itertools
import collections
import numpy as np

from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import EuclideanDistance
from nearpy.storage import MemoryStorage
from itertools import islice, izip, izip_longest, chain
from nearpy.utils import chunk
from nearpy.data import NumpyData
from collections import defaultdict
from time import time


class Engine(object):
    """
    Objects with this type perform the actual ANN search and vector indexing.
    They can be configured by selecting implementations of the Hash, Distance,
    Filter and Storage interfaces.

    There are four different modes of the engine:

        (1) Full configuration - All arguments are defined.
                In this case the distance and vector filters
                are applied to the bucket contents to deliver the
                resulting list of filtered (vector, data, distance) tuples.
        (2) No distance - The distance argument is None.
                In this case only the vector filters are applied to
                the bucket contents and the result is a list of
                filtered (vector, data) tuples.
        (3) No vector filter - The vector_filter argument is None.
                In this case only the distance is applied to
                the bucket contents and the result is a list of
                unsorted/unfiltered (vector, data, distance) tuples.
        (4) No vector filter and no distance - Both arguments are None.
                In this case the result is just the content from the
                buckets as an unsorted/unfiltered list of (vector, data)
                tuples.
    """

    def __init__(self, lshashes=None, distance=None, filters=[], storage=None):
        self.lshashes = lshashes
        self.distance = distance
        self.filters = filters

        self.storage = storage
        if self.storage is None:
            self.storage = MemoryStorage()

    # def set_metadata(self, attribute, metadata):
    #     for lshash in self.lshashes:
    #         self.storage.add_attribute(lshash.name, attribute)
    #         self.storage.set_metadata(lshash.name + "_" + attribute, metadata)

    def store(self, v, **datum):
        """
        Parameters
        ----------
        v: ndarray
            will be used to generate an hash key
        datum: JSON-serializable object
            will be stored in a bucket using the hash key

        Returns
        -------
        count:
            number of elements stored
        """
        V = np.array([v])
        data = {k: np.array([v]) for k, v in datum.items()}
        return self.store_batch(V, **data)

    def store_batch(self, V, data={}):
        """
        Parameters
        ----------
        V: iterable of ndarrays
            each ndarray will be used to generate an hash key
        data: iterable of JSON-serializable object
            each datum will be stored in a bucket using the hash key

        Returns
        -------
        count:
            number of elements stored
        """
        lshash = self.lshashes[0]

        # position = [v for k, v in data.items() if k.name == "position"]
        # if len(position) == 1:
        #     bucketkeys = lshash.hash_vector_with_pos(V, position[0])
        # else:
        start = time()
        bucketkeys = lshash.hash_vector(V)
        print "hashing: {:.2f}".format(time()-start)

        data[NumpyData("patch", V.dtype, V.shape[1:])] = V
        self.storage.store(bucketkeys, data)
        return bucketkeys

    def store_batch_with_pos(self, V, positions, data={}):
        """
        Parameters
        ----------
        V: iterable of ndarrays
            each ndarray will be used to generate an hash key
        data: iterable of JSON-serializable object
            each datum will be stored in a bucket using the hash key

        Returns
        -------
        count:
            number of elements stored
        """
        lshash = self.lshashes[0]
        print "Hashing codes..."
        start = time()
        bucketkeys = lshash.hash_vector_with_pos(V, positions)
        print "Codes hashed in {:.2f} sec.".format(time()-start)

        data[NumpyData("patch", V.dtype, V.shape[1:])] = V

        print "Storing..."
        start = time()
        self.storage.store(bucketkeys, data)
        print "Stored in {:.2f} sec.".format(time()-start)
        return bucketkeys

    def neighbors(self, v, *attributes):
        """
        Hashes vector v, collects all candidate vectors from the matching
        buckets in storage, applys the (optional) distance function and
        finally the (optional) filter function to construct the returned list
        of either (vector, data, distance) tuples or (vector, data) tuples.

        Parameters
        ----------
        v: ndarray
            will be used to generate an hash key

        Return
        ------
        candidates: list of tuples
            neighbors of `v`
        """
        V = np.array([v])
        neighbors = self.neighbors_batch(V, *attributes)
        return {k: v[0] for k, v in neighbors.items()}

    def neighbors_batch(self, V, *attributes):
        if self.distance is not None:
            if self.distance.attribute not in attributes:
                attributes += (self.distance.attribute,)

        start = time()
        lshash = self.lshashes[0]
        bucketkeys = lshash.hash_vector(V)
        print "Hashing: {:.2f}".format(time()-start)

        start = time()
        # Fetch only buckets that are unique
        bucketkeys, patch2bucket_indices = np.unique(bucketkeys, return_inverse=True)
        bucket2patch_indices = defaultdict(lambda: [])
        for i, idx in enumerate(patch2bucket_indices):
            bucket2patch_indices[idx] += [i]

        print "Uniquifying: {:.2f}".format(time()-start)

        #bucketcounts = np.array(self.storage.count(bucketkeys))
        #indices_sorted = np.argsort(bucketcounts)[::-1]
        #sorted_bucketcounts = bucketcounts[indices_sorted]

        neighborhood_filter = self.filters[0]

        buckets = {}
        start = time()
        for i, bucketkey in enumerate(bucketkeys):
            if i % 10 == 0:
                print "{:,}/{:,} ({:.2f} sec.)".format(i, len(bucketkeys), time()-start)
                start = time()

            #start = time()
            for attribute in attributes:
                buckets[attribute.name] = self.storage.retrieve([bucketkey], attribute)[0]

            #print "\nFetching: {:.2f}".format(time()-start)

            #start_loop = time()
            for j, patch_id in enumerate(bucket2patch_indices[i]):
                neighbors = {}

                # Distance
                neighbors['dist'] = self.distance(V[patch_id], buckets[self.distance.attribute.name])
                #print "Distance: {:.2f}".format(time()-start)

                # Filter
                #start = time()
                indices_to_keep = list(neighborhood_filter(neighbors['dist']))
                neighbors['dist'] = neighbors['dist'][indices_to_keep]
                #print "Filterin: {:.2f}".format(time()-start)

                for attribute in attributes:
                    neighbors[attribute.name] = buckets[attribute.name][indices_to_keep]

                yield patch_id, neighbors
            #print "Looping:  {:.2f} ({} x {})".format(time()-start_loop, len(bucket2patch_indices[i]), len(buckets[self.distance.attribute.name]))


        # nb_neighbors = 0
        # start = 0
        # end = 0
        # while start < len(bucketcounts):
        #     nb_neighbors = 0
        #     while nb_neighbors < 50000:
        #         end += 1
        #         nb_neighbors += np.sum(sorted_bucketcounts[start:end])

        #     bucket_indices = indices_sorted[start:end]
        #     bucketkeys_part = bucketkeys[bucket_indices]

        #     from time import time
        #     start = time()
        #     buckets = {}
        #     neighbors = {}
        #     for attribute in attributes:
        #         neighbors[attribute.name] = []
        #         buckets[attribute.name] = self.storage.retrieve(bucketkeys_part, attribute)

        #     print "Fetched in", time() - start

        #     for patch_id, idx in enumerate(patch2bucket_indices):
        #         neighbors = {}
        #         for attribute in attributes:
        #             neighbors[attribute.name] = buckets[attribute.name][i][list(neighbors_indices[i])])

        #         yield patch_id, neighbors[]


        #     start = time()
        #     if self.distance is not None:
        #         neighbors['dist'] = []
        #         for i, bucket_idx in enumerate(bucket_indices):
        #             bucket_patches = buckets[self.distance.attribute.name][i]

        #             for query_patch, idx in izip(V, patch2bucket_indices):
        #                 if idx != bucket_idx:
        #                     continue

        #                 neighbors['dist'].append(self.distance(query_patch, bucket_patches))

        #     print "Distance computed in", time() - start

        #     neighbors_indices = []
        #     for i, bucket_idx in enumerate(bucket_indices):
        #         for idx in patch2bucket_indices:
        #             if idx == bucket_idx:
        #                 neighbors_indices.append(xrange(len(buckets[attributes[0].name][i])))

        #     start = time()
        #     for neighborhood_filter in self.filters:
        #         for i, (neighbors_dist) in enumerate(neighbors["dist"]):
        #             neighbors_to_keep = neighborhood_filter(neighbors_dist)
        #             neighbors_indices[i] = neighbors_to_keep
        #             neighbors["dist"][i] = neighbors_dist[neighbors_to_keep]

        #     print "Filtered computed in", time() - start

        #     for attribute in attributes:
        #         for i, bucket_idx in enumerate(bucket_indices):
        #             for patch_id, idx in enumerate(patch2bucket_indices):
        #                 if idx == bucket_idx:
        #                     neighbors[attribute.name].append(buckets[attribute.name][i][list(neighbors_indices[i])])

        #     for patch_id, idx in enumerate(patch2bucket_indices):
        #         neighbors = {}
        #         for attribute in attributes:
        #             neighbors[attribute.name] = buckets[attribute.name][i][list(neighbors_indices[i])])

        #         yield patch_id, neighbors[]

    def neighbors_batch_with_pos(self, V, positions, radius, *attributes):
        if self.distance is not None:
            if self.distance.attribute not in attributes:
                attributes += (self.distance.attribute,)

        lshash = self.lshashes[0]
        bucketkeys = []
        nb_query_patches = len(V)
        #nb_spatial_neighbors = (2*radius+1)**3
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                for z in range(-radius, radius+1):
                    bucketkeys += lshash.hash_vector_with_pos(V, positions + np.array([x, y, z]))

        # Fetch only buckets that are unique
        bucketkeys, patch2bucket_indices = np.unique(bucketkeys, return_inverse=True)
        bucket2patch_indices = defaultdict(lambda: [])
        for i, idx in enumerate(patch2bucket_indices):
            bucket2patch_indices[idx] += [i]

        neighborhood_filter = self.filters[0]

        buckets = {}
        for i, bucketkey in enumerate(bucketkeys):
            if i % 1000 == 0:
                print "{:,}/{:,}".format(i, len(bucketkeys))

            for attribute in attributes:
                buckets[attribute.name] = self.storage.retrieve([bucketkey], attribute)[0]

            for j, patch_id in enumerate(bucket2patch_indices[i]):
                neighbors = {}

                # Distance
                neighbors['dist'] = self.distance(V[patch_id % nb_query_patches], buckets[self.distance.attribute.name])

                # Filter
                indices_to_keep = list(neighborhood_filter(neighbors['dist']))
                neighbors['dist'] = neighbors['dist'][indices_to_keep]

                for attribute in attributes:
                    neighbors[attribute.name] = buckets[attribute.name][indices_to_keep]

                yield patch_id % nb_query_patches, neighbors

    # def neighbors_batch(self, V, *attributes):
    #     lshash = self.lshashes[0]
    #     bucketkeys = lshash.hash_vector(V)

    #     # Fetch only buckets that are unique
    #     bucketkeys, indices = np.unique(bucketkeys, return_inverse=True)

    #     if self.distance is not None:
    #         if self.distance.attribute not in attributes:
    #             attributes += (self.distance.attribute,)

    #     from time import time
    #     start = time()
    #     buckets = {}
    #     neighbors = {}
    #     for attribute in attributes:
    #         neighbors[attribute.name] = []
    #         buckets[attribute.name] = self.storage.retrieve(bucketkeys, attribute)

    #     print "Fetched in", time() - start

    #     start = time()
    #     if self.distance is not None:
    #         neighbors['dist'] = []
    #         for i, (query_patch, idx) in enumerate(izip(V, indices)):
    #             bucket_patches = buckets[self.distance.attribute.name][idx]
    #             neighbors['dist'].append(self.distance(query_patch, bucket_patches))

    #     print "Distance computed in", time() - start

    #     neighbors_indices = [xrange(len(buckets[attributes[0].name][idx])) for idx in indices]

    #     start = time()
    #     for neighborhood_filter in self.filters:
    #         for i, (neighbors_dist, idx) in enumerate(izip(neighbors["dist"], indices)):
    #             neighbors_to_keep = neighborhood_filter(neighbors_dist)
    #             neighbors_indices[i] = neighbors_to_keep
    #             neighbors["dist"][i] = neighbors_dist[neighbors_to_keep]

    #     print "Filtered computed in", time() - start

    #     for attribute in attributes:
    #         for i, idx in enumerate(indices):
    #             neighbors[attribute.name].append(buckets[attribute.name][idx][list(neighbors_indices[i])])

    #     return neighbors

    def neighbors_batch_extended(self, V, min_neighbors=1, *attributes):
        lshash = self.lshashes[0]
        bucketkeys = lshash.hash_vector(V)

        if self.distance is not None:
            if self.distance.attribute not in attributes:
                attributes += (self.distance.attribute,)

        # neighbors = {}
        # for attribute in attributes:
        #     neighbors[attribute.name] = self.storage.retrieve(bucketkeys, attribute)

        # for neighbors_patches in neighbors["patch"]:
        #     if len(neighbors_patches) < min_neighbors:


        # lshash = self.lshashes[0]
        # bucketkeys = lshash.hash_vector(V)

        # if self.distance is not None:
        #     if self.distance.attribute not in attributes:
        #         attributes += (self.distance.attribute,)

        # neighbors = {}
        # for attribute in attributes:
        #     neighbors[attribute.name] = self.storage.retrieve(bucketkeys, attribute)

        counts = np.array(self.storage.count(bucketkeys))
        print np.sum(counts == 0)
        from ipdb import set_trace as dbg
        dbg()

        for i in np.where(counts < min_neighbors)[0]:
            for j in range(len(bucketkeys[i])):
                key = bucketkeys[i]
                key = key[:1] + str(1-int(key[j])) + key[j+1:]
                if self.storage.count(bucketkeys)[0] > 0:
                    bucketkeys[i] = key

        counts2 = np.array(self.storage.count([bucketkey_prefix + key + "_patch" for key in hashkeys]))
        print np.sum(counts2 == 0)

        from ipdb import set_trace as dbg
        dbg()
        neighbors = {}
        for attribute in attributes:
            neighbors[attribute.name] = self.storage.retrieve(hashkeys, attribute, prefix=bucketkey_prefix)

        return neighbors

    def candidate_count(self, v):
        """
        Returns candidate count for nearest neighbour search for specified vector.
        The candidate count is the count of vectors taken from all buckets the
        specified vector is projected onto.

        Use this method to check if your hashes are configured good. High candidate
        counts makes querying slow.

        For example if you always want to retrieve 20 neighbours but the candidate
        count is 1000 or something you have to change the hash so that each bucket
        has less entries (increase projection count for example).

        Parameters
        ----------
        v: ndarray
            will be used to generate an hash key

        Return
        ------
        count: int
            candidate count for `v`
        """
        V = np.array([v])
        return self.candidate_count_batch(V)[0]

    def candidate_count_batch(self, V):
        """
        Returns candidate count for nearest neighbour search for specified vector.
        The candidate count is the count of vectors taken from all buckets the
        specified vector is projected onto.

        Use this method to check if your hashes are configured good. High candidate
        counts makes querying slow.

        For example if you always want to retrieve 20 neighbours but the candidate
        count is 1000 or something you have to change the hash so that each bucket
        has less entries (increase projection count for example).

        Parameters
        ----------
        V: iterable of ndarrays
            each ndarray will be used to generate an hash key

        Return
        ------
        counts: list of int
            candidate count of each element in `V`
        """
        lshash = self.lshashes[0]
        bucketkeys = lshash.hash_vector(V)
        counts = self.storage.count(bucketkeys)
        return counts

    # def targets_count(self):
    #     lshash = self.lshashes[0]
    #     return self.storage.retrieve("*", attribute="target", prefix=lshash.name + "_")
    #     # attribute_metadata = self.storage.get_metadata(lshash.name + "_target")
    #     # dtype = attribute_metadata['dtype']
    #     # shape = (-1,) + eval(attribute_metadata['shape'])
    #     # contents = self.storage.retrieve("*", attribute="target", prefix=lshash.name + "_")
    #     # targets = np.frombuffer("".join(chain(*contents)), dtype).reshape(shape)
    #     # return np.bincount(targets[:, 0])

    def buckets_size(self):
        bucketkeys = self.storage.bucketkeys()
        return self.storage.count(bucketkeys), bucketkeys

    def nb_patches(self):
        bucketkeys = self.storage.bucketkeys(as_generator=True)
        return sum(self.storage.count(bucketkeys))

    def nb_buckets(self):
        bucketkeys = self.storage.bucketkeys()
        return len(bucketkeys)

    def clean_all_buckets(self):
        """ Clears buckets in storage (removes all vectors and their data). """
        bucketkeys = self.storage.bucketkeys_all_attributes(as_generator=True)
        return self.storage.clear(bucketkeys)

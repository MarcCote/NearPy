# -*- coding: utf-8 -*-
from __future__ import print_function

# import json
# import itertools
# import collections
import numpy as np

#from nearpy.hashes import RandomBinaryProjections
#from nearpy.filters import NearestFilter
#from nearpy.distances import EuclideanDistance
from nearpy.storage import storage_factory
#from itertools import islice, izip, izip_longest, chain
from nearpy.utils import chunk, Timer
#from nearpy.data import NumpyData
from collections import defaultdict
from time import time


def flip(bits, no_bit):
    return np.bitwise_xor(np.uint64(bits), np.left_shift(np.uint64(1), np.uint64(no_bit)))


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
            self.storage = storage_factory("memory")

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
        with Timer("  Hashing"):
            bucketkeys = lshash.hash_vector(V)

        #data[NumpyData("patch", V.dtype, V.shape[1:])] = V
        bucketkeys = list(chunk(bucketkeys.tostring(), bucketkeys.itemsize))
        self.storage.store(bucketkeys, data)
        return bucketkeys

    # def store_batch_with_pos(self, V, positions, data={}):
    #     """
    #     Parameters
    #     ----------
    #     V: iterable of ndarrays
    #         each ndarray will be used to generate an hash key
    #     data: iterable of JSON-serializable object
    #         each datum will be stored in a bucket using the hash key

    #     Returns
    #     -------
    #     count:
    #         number of elements stored
    #     """
    #     lshash = self.lshashes[0]
    #     print "Hashing codes..."
    #     start = time()
    #     bucketkeys = lshash.hash_vector_with_pos(V, positions)
    #     print "Codes hashed in {:.2f} sec.".format(time()-start)

    #     data[NumpyData("patch", V.dtype, V.shape[1:])] = V

    #     print "Storing..."
    #     start = time()
    #     self.storage.store(bucketkeys, data)
    #     print "Stored in {:.2f} sec.".format(time()-start)
    #     return bucketkeys

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

    def neighbors_batch(self, V, patches, *attributes):
        if self.distance is not None:
            if self.distance.attribute not in attributes:
                attributes += (self.distance.attribute,)

        start = time()
        lshash = self.lshashes[0]
        with Timer("  Hashing"):
            bucketkeys = lshash.hash_vector(V)

        with Timer("  Uniquifying"):
            # Fetch only buckets that are unique
            unique_bucketkeys, patch2bucket_indices = np.unique(bucketkeys, return_inverse=True)
            #unique_bucketkeys, patch2bucket_indices = np.unique(bucketkeys.view("int64"), return_inverse=True)
            #unique_bucketkeys = unique_bucketkeys.view("|S8")

            bucket2patch_indices = defaultdict(lambda: [])
            for i, idx in enumerate(patch2bucket_indices):
                bucket2patch_indices[idx] += [i]

        # with Timer("  Counting"):
        #     bucketcounts = np.array(self.storage.count(unique_bucketkeys))

        # indices_sorted = np.argsort(bucketcounts)[::-1]
        # sorted_bucketcounts = bucketcounts[indices_sorted]
        #indices_sorted = np.argsort(bucketcounts)[::-1]
        #sorted_bucketcounts = bucketcounts[indices_sorted]

        #import theano
        #import theano.tensor as T
        # Patches = theano.shared(np.zeros((1, 9), dtype="float32"))
        # query = T.vector()
        # distances = T.sqrt(T.mean((Patches - query) ** 2, axis=1))
        # f_dist = theano.function([query], T.argsort(distances)[:100])

        neighborhood_filter = self.filters[0]

        buckets = {}
        start = time()
        for i, bucketkey in enumerate(chunk(unique_bucketkeys.tostring(), unique_bucketkeys.itemsize)):
            if i % 1000 == 0:
                print("{:,}/{:,} ({:.2f} sec.)".format(i, len(unique_bucketkeys), time()-start))
                start = time()

            #with Timer("  Fetching"):
            for attribute in attributes:
                buckets[attribute.name] = self.storage.retrieve([bucketkey], attribute)[0]

            # TODO: Generalize to more than one bit flipping
            # Check if we have enough neighbors
            # no_bit = 0
            # while len(buckets[self.distance.attribute.name]) < neighborhood_filter.K and no_bit < lshash.nbits:
            #     print("Flipping bit:", no_bit)
            #     newkey = flip(np.fromstring(bucketkey, dtype=np.uint64), no_bit).tostring()
            #     for attribute in attributes:
            #         buckets[attribute.name] = np.r_[buckets[attribute.name],
            #                                         self.storage.retrieve([newkey], attribute)[0]]
            #     no_bit += 1

            #with Timer("  SubFetching"):
            if len(buckets[self.distance.attribute.name]) < neighborhood_filter.K:
                newkeys_str = flip(np.fromstring(bucketkey, dtype=np.uint64), range(lshash.nbits)).tostring()
                newkeys = list(chunk(newkeys_str, unique_bucketkeys.itemsize))

                for attribute in attributes:
                    buckets[attribute.name] = np.vstack([buckets[attribute.name]] + self.storage.retrieve(newkeys, attribute))

            #Patches.set_value(buckets[self.distance.attribute.name].reshape((-1, 9)))
            #start_loop = time()
            for j, patch_id in enumerate(bucket2patch_indices[i]):
                neighbors = {}

                # Distance
                #with Timer("    Distance "):
                neighbors['dist'] = self.distance(patches[patch_id], buckets[self.distance.attribute.name])

                # Filter
                #with Timer("    Filtering"):
                indices_to_keep = list(neighborhood_filter(neighbors['dist']))
                #indices_to_keep = f_dist(patches[patch_id].flatten())
                neighbors['dist'] = neighbors['dist'][indices_to_keep]

                for attribute in attributes:
                    neighbors[attribute.name] = buckets[attribute.name][indices_to_keep]

                yield patch_id, neighbors
            #print "Looping:  {:.2f} ({} x {})".format(time()-start_loop, len(bucket2patch_indices[i]), len(buckets[self.distance.attribute.name]))

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
                print("{:,}/{:,}".format(i, len(bucketkeys)))

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
        return self.storage.remove(bucketkeys)

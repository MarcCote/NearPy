from __future__ import print_function

import types
import shutil

import rocksdb
import os

from collections import defaultdict
from itertools import izip, takewhile
from nearpy.storage.storage import Storage
from os.path import join as pjoin
from nearpy.utils.utils import Timer

PREFIX_LENGTH = 10


class BucketMerger(rocksdb.interfaces.AssociativeMergeOperator):
    def merge(self, key, existing_value, value):
        if existing_value:
            return (True, existing_value+value)
        return (True, value)

    def name(self):
        return b'BucketMerger'


class AttributePrefix(rocksdb.interfaces.SliceTransform):
    def name(self):
        return b'attribute'

    def transform(self, src):
        return (0, PREFIX_LENGTH)

    def in_domain(self, src):
        return len(src) >= PREFIX_LENGTH

    def in_range(self, dst):
        return len(dst) == PREFIX_LENGTH and dst[-1] == b":"


class RocksDBStorage(Storage):
    """ Storage using RocksDB. """

    def __init__(self, name, root="./"):
        self.db_filename = pjoin(root, name)

        #Create repository structure
        if not os.path.isdir(root):
            os.makedirs(root)

        options = rocksdb.Options()
        options.create_if_missing = True
        options.max_open_files = 500  # Default on Unix is ~1024
        options.compression = rocksdb.CompressionType.snappy_compression
        options.merge_operator = BucketMerger()
        options.prefix_extractor = AttributePrefix()

        with Timer("Opening RocksDB: {}".format(name)):
            self.db = rocksdb.DB(self.db_filename, options)

    def store(self, bucketkeys, bucketvalues):
        buckets = defaultdict(lambda: [])
        with Timer("  Bucketing"):
            for attribute, values in bucketvalues.items():
                prefix = str(attribute.name.ljust(PREFIX_LENGTH) + b":")
                for key, value in izip(bucketkeys, attribute.dumps(values)):
                    buckets[prefix + key].append(value)

        batch = rocksdb.WriteBatch()
        with Timer("  Batching"):
            for key, value in buckets.items():
                batch.merge(key, b"".join(value))

        with Timer("  Writing"):
            self.db.write(batch, sync=True)

        return len(bucketkeys)

    def retrieve(self, bucketkeys, attribute):
        prefix = str(attribute.name.ljust(PREFIX_LENGTH) + b":")
        keys = [prefix + bucketkey for bucketkey in bucketkeys]

        #with Timer("  Retrieving"):
        results = self.db.multi_get(keys)
        return [attribute.loads(result) for result in results.values()]

    def remove(self, bucketkeys):
        """
        Parameters
        ----------
        bucket_keys: iterable of string
            keys of the buckets to delete
        prefix: string
            if set, clear every buckets having this prefix

        Return
        ------
        count: int
            number of buckets cleared
        """
        if (not isinstance(bucketkeys, types.ListType) and not isinstance(bucketkeys, types.GeneratorType)
                and not hasattr(bucketkeys, "__iter__")):
            bucketkeys = [bucketkeys]

        count = 0
        batch = rocksdb.WriteBatch()
        with Timer("  Batching"):
            for bucketkey in bucketkeys:
                if self.db.key_may_exist(bucketkey)[0]:
                    batch.delete(bucketkey)
                    count += 1

        with Timer("  Writing"):
            self.db.write(batch, sync=True)
            self.db.compact_range()

        return count

    def count(self, bucketkeys):
        """
        Parameters
        ----------
        bucketkeys: iterable of string
            keys of buckets to count

        Return
        ------
        counts: list of int
            size of each given bucket
        """
        prefix = "label".ljust(PREFIX_LENGTH) + b":"
        counts = []

        items = self.db.iteritems()
        items.seek(prefix)
        items = takewhile(lambda e: e[0].startswith(prefix), items)

        for k, v in items:
            counts.append(len(v))  # We suppose each label fits in a byte.

        return counts

    def bucketkeys(self, deprecated_pattern=".*", as_generator=False):
        prefix = "id".ljust(PREFIX_LENGTH) + b":"
        keys = self.db.iterkeys()
        keys.seek(prefix)
        keys = takewhile(lambda k: k.startswith(prefix), keys)
        keys = (k[len(prefix):] for k in keys)  # Remove prefix

        return keys if as_generator else list(keys)

    def bucketkeys_all_attributes(self, deprecated_pattern=".*", as_generator=False):
        keys = self.db.iterkeys()
        keys.seek_to_first()
        return keys if as_generator else list(keys)

    def clear(self):
        del self.db
        shutil.rmtree(self.db_filename)

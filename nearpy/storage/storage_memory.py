import re
import types

from collections import defaultdict
from itertools import izip
from nearpy.storage.storage import Storage


class MemoryStorage(Storage):
    """ Storage in memory. """

    def __init__(self, keyprefix=""):
        self.infos = defaultdict(lambda: [])
        self.buckets = defaultdict(lambda: [])
        self.keyprefix = keyprefix

    def get_info(self, key):
        return self.infos[key]

    def set_info(self, key, value, append=False):
        if append:
            self.infos[key].append(value)
        else:
            self.infos[key] = value

    def del_info(self, key, value=None):
        if value is not None:
            self.infos[key].remove(value)
        else:
            del self.infos[key]

    def store(self, bucketkeys, bucketvalues):
        keys = [self.keyprefix + "_" + bucketkey for bucketkey in bucketkeys]
        for attribute, values in bucketvalues.items():
            for key, value in izip(keys, attribute.dumps(values)):
                self.buckets[key + "_" + attribute.name].append(value)

        return len(bucketkeys)

    def retrieve(self, bucketkeys, attribute):
        keys = [self.keyprefix + "_" + bucketkey + '_' + attribute.name for bucketkey in bucketkeys]
        results = [self.credis.execute("lrange", key, 0, -1) for key in keys]
        return [attribute.loads("".join(result)) for result in results]

    def clear(self, bucketkeys):
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
        if not isinstance(bucketkeys, types.ListType) and not isinstance(bucketkeys, types.GeneratorType):
            bucketkeys = [bucketkeys]

        count = 0
        for bucketkey in bucketkeys:
            key = self.keyprefix + "_" + bucketkey
            if key in self.buckets:
                del self.buckets[key]
                count += 1

        return count

    def count(self, bucketkeys):
        """
        Parameters
        ----------
        bucketkeys: iterable of string
            keys from which to retrieve values

        Return
        ------
        counts: list of int
            size of each given bucket
        """
        counts = []
        suffix = "_patch"
        for bucketkey in bucketkeys:
            key = self.keyprefix + "_" + bucketkey + suffix
            counts.append(len(self.buckets[key]))

        return counts

    def bucketkeys(self, pattern=".", as_generator=False):
        suffix = "patch"
        pattern = "{prefix}_{pattern}_{suffix}".format(prefix=self.keyprefix, pattern=pattern, suffix=suffix)
        regex = re.compile(pattern)
        start = len(self.keyprefix) + 1
        end = -(len(suffix) + 1)

        keys = (key[start:end] for key in self.buckets.keys() if regex.match(key) is not None)
        if not as_generator:
            keys = list(keys)

        return keys

    def bucketkeys_all_attributes(self, pattern=".", as_generator=False):
        pattern = "{prefix}_{pattern}".format(prefix=self.keyprefix, pattern=pattern)
        regex = re.compile(pattern)
        start = len(self.keyprefix) + 1

        keys = (key[start:] for key in self.buckets.keys() if regex.match(key) is not None)
        if not as_generator:
            keys = list(keys)

        return keys

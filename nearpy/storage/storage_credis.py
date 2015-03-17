import json
import redis
import credis
import types

from itertools import izip, chain
from nearpy.storage.storage import Storage
from nearpy.utils import chunk, ichunk

BUFFER_SIZE = 100000


class CRedisStorage(Storage):
    """ Storage using credis. """

    def __init__(self, host='localhost', port=6379, db=0, keyprefix=""):
        """ Uses specified redis object for storage. """
        self.credis = credis.Connection(host=host, port=port)
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.keyprefix = keyprefix
        self.infos_key = "infos"

    @property
    def infos(self):
        data = self.redis.get(self.infos_key)
        if data is None:
            return {}

        return json.loads(data)

    @infos.setter
    def infos(self, value):
        self.redis.set(self.infos_key, json.dumps(value))

    def get_info(self, key):
        return self.infos.get(key, [])

    def set_info(self, key, value, append=False):
        infos = self.infos

        if append:
            if key not in infos:
                infos[key] = []
            infos[key].append(value)
        else:
            infos[key] = value

        self.infos = infos

    def del_info(self, key, value=None):
        infos = self.infos

        if value is not None:
            infos[key].remove(value)
        else:
            del infos[key]

        self.infos = infos

    def store(self, bucketkeys, bucketvalues):
        keys = [self.keyprefix + "_" + bucketkey for bucketkey in bucketkeys]
        commands = []
        for attribute, values in bucketvalues.items():
            for key, value in izip(keys, attribute.dumps(values)):
                commands.append(("rpush", key + "_" + attribute.name, value))
                #self.credis.execute("rpush", key + "_" + attribute.name, value)

        self.credis.execute_pipeline(*commands)
        return len(bucketkeys)

    def retrieve_batch(self, bucketkeys, attribute):
        keys = [self.keyprefix + "_" + bucketkey + '_' + attribute.name for bucketkey in bucketkeys]
        commands = [("lrange", key, 0, -1) for key in keys]
        results = self.credis.execute_pipeline(*commands)

        # for bucketkeys_chunk in ichunk(bucketkeys, n=BUFFER_SIZE):
        #     keys = [self.keyprefix + "_" + bucketkey + '_' + attribute.name for bucketkey in bucketkeys_chunk]
        #     commands = [("lrange", key, 0, -1) for key in keys]
        #     results += self.credis.execute_pipeline(*commands)

        return [attribute.loads("".join(result)) for result in results]

    def retrieve(self, bucketkeys, attribute):
        keys = [self.keyprefix + "_" + bucketkey + '_' + attribute.name for bucketkey in bucketkeys]
        results = [self.credis.execute("lrange", key, 0, -1) for key in keys]
        return [attribute.loads("".join(result)) for result in results]

    def retrieve_all(self, bucketkeys, attribute):
        results = []
        for bucketkeys_chunk in ichunk(bucketkeys, n=BUFFER_SIZE):
            keys = [self.keyprefix + "_" + bucketkey + '_' + attribute.name for bucketkey in bucketkeys_chunk]
            commands = [("lrange", key, 0, -1) for key in keys]
            results += self.credis.execute_pipeline(*commands)

        return attribute.loads("".join(chain(*results)))

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
        for bucketkeys_chunk in ichunk(bucketkeys, n=BUFFER_SIZE):
            keys = [self.keyprefix + "_" + bucketkey for bucketkey in bucketkeys_chunk]
            count += self.credis.execute("del", *keys)

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
        for bucketkeys_chunk in ichunk(bucketkeys, n=BUFFER_SIZE):
            commands = [("llen", self.keyprefix + "_" + bucketkey + suffix) for bucketkey in bucketkeys_chunk]
            counts += self.credis.execute_pipeline(*commands)

        return counts

    # def bucketkeys(self, pattern, prefix=None):
    #     """
    #     Parameters
    #     ----------
    #     prefix: string
    #         if set, report counts of every buckets having this prefix
    #     """
    #     if prefix is None:
    #         prefix = self.keyprefix

    #     return [key[len(prefix):] for key in self.credis.execute("keys", prefix + pattern)]

    def bucketkeys(self, pattern="*", as_generator=False):
        suffix = "patch"
        pattern = "{prefix}_{pattern}_{suffix}".format(prefix=self.keyprefix, pattern=pattern, suffix=suffix)
        start = len(self.keyprefix) + 1
        end = -(len(suffix) + 1)

        if as_generator:
            keys = (key[start:end] for key in self.redis.scan_iter(match=pattern, count=BUFFER_SIZE))
        else:
            keys = [key[start:end] for key in self.credis.execute("keys", pattern)]

        return keys

    def bucketkeys_all_attributes(self, pattern="*", as_generator=False):
        pattern = "{prefix}_{pattern}".format(prefix=self.keyprefix, pattern=pattern)
        start = len(self.keyprefix) + 1

        if as_generator:
            keys = (key[start:] for key in self.redis.scan_iter(match=pattern, count=BUFFER_SIZE))
        else:
            keys = [key[start:] for key in self.credis.execute("keys", pattern)]

        return keys

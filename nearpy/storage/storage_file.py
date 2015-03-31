import re
import types

import os
from time import time

from collections import defaultdict
from itertools import izip
from nearpy.storage.storage import Storage
from os.path import join as pjoin
from nearpy.utils.utils import load_dict_from_json, save_dict_to_json


class FileStorage(Storage):
    """ Storage using files and folders. """

    def __init__(self, name, root="./"):
        self.root = root
        self.infos_filename = pjoin(root, "infos.json")

        #Create repository structure
        if not os.path.isdir(root):
            os.makedirs(root)

        if name != "":
            self.buckets_dir = pjoin(root, name)
            if not os.path.isdir(self.buckets_dir):
                os.makedirs(self.buckets_dir)

        if not os.path.isfile(self.infos_filename):
            save_dict_to_json(self.infos_filename, {})

    def get_info(self, key):
        infos = load_dict_from_json(self.infos_filename)
        return infos.get(key, [])

    def set_info(self, key, value, append=False):
        infos = load_dict_from_json(self.infos_filename)

        if append:
            if key not in infos:
                infos[key] = []
            infos[key].append(value)
        else:
            infos[key] = value

        save_dict_to_json(self.infos_filename, infos)

    def del_info(self, key, value=None):
        infos = load_dict_from_json(self.infos_filename)

        if value is not None:
            infos[key].remove(value)
        else:
            del infos[key]

        save_dict_to_json(self.infos_filename, infos)

    def store(self, bucketkeys, bucketvalues):
        buf = defaultdict(lambda: [])
        start = time()
        for attribute, values in bucketvalues.items():
            for key, value in izip(bucketkeys, attribute.dumps(values)):
                filename = pjoin(self.buckets_dir, key + "_" + attribute.name + ".npy")
                buf[filename].append(value)

        print "buffering: {:.2f} ({:,} buckets)".format(time()-start, len(buf)/len(bucketvalues))

        start = time()
        for filename, values in buf.items():
            # with open(filename, 'ab') as f:
            #     f.write("".join(values))

            data = ""
            if os.path.isfile(filename):
                data = open(filename, 'rb').read()
            open(filename, 'wb').write(data + "".join(values))

        print "writing: {:.2f}".format(time()-start)
        return len(bucketkeys)

    def retrieve(self, bucketkeys, attribute):
        filenames = [pjoin(self.buckets_dir, bucketkey + "_" + attribute.name + ".npy") for bucketkey in bucketkeys]

        results = []
        for filename in filenames:
            if os.path.isfile(filename):
                results.append(open(filename).read())
            else:
                results.append("")

        return [attribute.loads("".join(result)) for result in results]

    def clear(self, bucketkeys):
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
        if not isinstance(bucketkeys, types.ListType) and not isinstance(bucketkeys, types.GeneratorType):
            bucketkeys = [bucketkeys]

        count = 0
        for bucketkey in bucketkeys:
            filename = pjoin(self.buckets_dir, bucketkey + ".npy")
            if os.path.isfile(filename):
                os.remove(filename)
                count += 1

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
        counts = []
        suffix = "_label"
        for bucketkey in bucketkeys:
            filename = pjoin(self.buckets_dir, bucketkey + suffix + ".npy")
            nb_bytes = os.path.getsize(filename)
            counts.append(nb_bytes)  # We suppose each label fits in a byte.

        return counts

    def bucketkeys(self, pattern=".*", as_generator=False):
        suffix = "patch"
        extension = ".npy"
        pattern = "{pattern}_{suffix}{extension}".format(pattern=pattern, suffix=suffix, extension=extension)
        regex = re.compile(pattern)
        end = -(len(suffix) + len(extension) + 1)

        filenames = os.listdir(self.buckets_dir)
        keys = (filename[:end] for filename in filenames if regex.match(filename) is not None)
        if not as_generator:
            keys = list(keys)

        return keys

    def bucketkeys_all_attributes(self, pattern=".*", as_generator=False):
        extension = ".npy"
        pattern = "{pattern}{extension}".format(pattern=pattern, extension=extension)
        regex = re.compile(pattern)
        end = -len(extension)

        filenames = os.listdir(self.buckets_dir)
        keys = (filename[:end] for filename in filenames if regex.match(filename) is not None)
        if not as_generator:
            keys = list(keys)

        return keys

# -*- coding: utf-8 -*-
from __future__ import absolute_import


def storage_factory(name, keyprefix="", **kwargs):
    if name.lower() == "memory":
        from nearpy.storage.storage_memory import MemoryStorage
        storage = MemoryStorage(keyprefix=keyprefix)
    elif name.lower() == "file":
        from nearpy.storage.storage_file import FileStorage
        storage = FileStorage(keyprefix=keyprefix,
                              dir=kwargs.get("dir", "./db"))
    elif name.lower() == "redis":
        from nearpy.storage.storage_credis import CRedisStorage
        storage = CRedisStorage(keyprefix=keyprefix,
                                host=kwargs.get("host", "localhost"),
                                port=kwargs.get("port", 6379),
                                db=kwargs.get('db', 0))
    elif name.lower() == "rocksdb":
        from nearpy.storage.storage_rocksdb import RocksDBStorage
        storage = RocksDBStorage(name=keyprefix,
                                 root=kwargs.get("dir", "./db"))
    else:
        raise ValueError("Unknown storage: {}".format(name))

    storage.name = name.lower()
    return storage

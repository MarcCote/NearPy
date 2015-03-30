# -*- coding: utf-8 -*-


class Storage(object):
    """ Interface for storage adapters. """

    def store_vector(self, hash_name, bucket_key, v, data):
        """
        Stores vector and JSON-serializable data in bucket with specified key.
        """
        raise NotImplementedError

    def get_bucket(self, hash_name, bucket_key):
        """
        Returns bucket content as list of tuples (vector, data).
        """
        raise NotImplementedError

    def clean_buckets(self, hash_name):
        """
        Removes all buckets and their content.
        """
        raise NotImplementedError

    def clean_all_buckets(self):
        """
        Removes all buckets and their content.
        """
        raise NotImplementedError

    def store_hash_configuration(self, lshash):
        """
        Stores hash configuration
        """
        raise NotImplementedError

    def load_hash_configuration(self, hash_name):
        """
        Loads and returns hash configuration
        """
        raise NotImplementedError

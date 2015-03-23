import numpy as np
import pickle
from nearpy.hashes import SpectralHashing

ts = 3
X = np.array([[1, 100, -5], [4, 0, 21], [3.2, -10, -7], [3, 17, 13]], dtype="float32")
mx = X.max(axis=0)+1e-16
mn = X.min(axis=0)-1e-16

pickle.dump((np.zeros(3, dtype="float32"), (np.ones(3, dtype="float32"), np.eye(3).astype("float32"))), open('/tmp/test_pca.pkl', 'w'))

sh = SpectralHashing(name="test", dimension=3, trainset=lambda: iter([X]), nbits=3, pkl="/tmp/test_pca.pkl")
print sh.bounds
print mn, mx

print sh.modes
# for n_bits=3, should print
# [[ 0.  1.  0.]
#  [ 0.  2.  0.]
#  [ 0.  3.  0.]]

sh = SpectralHashing(name="test", dimension=3, trainset=lambda: iter([X]), nbits=6, pkl="/tmp/test_pca.pkl")
print sh.modes
# for n_bits = 6, should print
#[[ 0.  1.  0.]
# [ 0.  2.  0.]
# [ 0.  3.  0.]
# [ 0.  0.  1.]
# [ 0.  4.  0.]
# [ 0.  5.  0.]]

print sh.hash_vector([mn]).view('uint'), "==", 63
print sh.hash_vector([mx]).view('uint'), "==", 18
print sh.hash_vector([0.7*mn+0.3*mx]).view('uint'), "==", 9
print sh.hash_vector([0.3*mn+0.7*mx]).view('uint'), "==", 4
# Codes should be
# 63
# 18
# 9
# 4

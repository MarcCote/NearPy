
import unittest

import nearpy.tests as tests

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestStorage)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestRandomBinaryProjections)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestRandomDiscretizedProjections)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestEngine)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestEuclideanDistance)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestCosineDistance)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestManhattanDistance)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestVectorFilters)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestPCABinaryProjections)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestPCADiscretizedProjections)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestHashStorage)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestRandomBinaryProjectionTree)
unittest.TextTestRunner(verbosity=2).run(suite)

# TODO: Experiment tests are out of date!
#suite = unittest.TestLoader().loadTestsFromTestCase(
#    tests.TestRecallExperiment)
#unittest.TextTestRunner(verbosity=2).run(suite)

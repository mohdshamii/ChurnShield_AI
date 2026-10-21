"""
ChurnShield AI - Unified Test Runner
Runs the full suite of unit and integration tests.
"""

import sys
import unittest

def run_all_tests():
    print("\n========================================================")
    print("[TESTS] Running ChurnShield AI Full Test Suite")
    print("========================================================\n")

    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        print(f"\n[FAIL] Test suite failed with {len(result.failures)} failures and {len(result.errors)} errors.")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] All {result.testsRun} tests passed successfully!")
        sys.exit(0)

if __name__ == '__main__':
    run_all_tests()

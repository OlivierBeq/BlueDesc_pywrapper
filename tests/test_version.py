"""Trivial version test."""

import unittest

import BlueDesc_pywrapper


class TestVersion(unittest.TestCase):
    """Trivially test a version."""

    def test_version_type(self):
        """Test the version is a string.

        This is only meant to be an example test.
        """
        version = BlueDesc_pywrapper.__version__
        self.assertIsInstance(version, str)


if __name__ == "__main__":
    unittest.main()

"""Tests for :mod:`BlueDesc_pywrapper.utils`."""

import os
import unittest

from rdkit import Chem

from BlueDesc_pywrapper.utils import (
    get_java_in_dir,
    install_java,
    mktempdir,
    mktempfile,
    needsHs,
)


class TestNeedsHs(unittest.TestCase):
    """Tests for needsHs."""

    def test_needs_hs_without_explicit_hydrogens(self):
        mol = Chem.MolFromSmiles("CCO")
        self.assertTrue(needsHs(mol))

    def test_does_not_need_hs_with_explicit_hydrogens(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.assertFalse(needsHs(mol))


class TestTempFileHelpers(unittest.TestCase):
    """Tests for mktempdir/mktempfile."""

    def test_mktempdir_creates_writeable_dir(self):
        path = mktempdir()
        self.addCleanup(os.rmdir, path)
        self.assertTrue(os.path.isdir(path))

    def test_mktempfile_creates_writeable_file(self):
        path = mktempfile(suffix="foo.txt")
        self.addCleanup(os.remove, path)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("foo.txt"))


class TestJavaInstall(unittest.TestCase):
    """Tests for install_java/get_java_in_dir."""

    def test_install_java_returns_existing_path(self):
        path = install_java()
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))

    def test_get_java_in_dir_missing_version_returns_none(self):
        search_dir = mktempdir()
        self.addCleanup(os.rmdir, search_dir)
        path = get_java_in_dir(search_dir, version=999)
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()

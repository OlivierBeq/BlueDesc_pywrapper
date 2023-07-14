# -*- coding: utf-8 -*-

"""Tests for molecular descriptors."""

import unittest

from BlueDesc_pywrapper import BlueDesc
from tests.constants import *


class TestDescriptors(unittest.TestCase):
    """Tests for CDK_pywrapper molecular descriptors."""

    def setUp(self) -> None:
        """Create the molecular descriptor calculator."""
        self.blu = BlueDesc()
        self.blu3d = BlueDesc(ignore_3D=False)
        self.molecules = list(MOLECULES.values())

    def test_2D_descriptor_size(self):
        values = self.blu.calculate(self.molecules, show_banner=False)
        self.assertEqual(values.shape, (len(MOLECULES), 118))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 118)

    def test_2D_descriptor_multithread(self):
        values = self.blu.calculate(self.molecules, show_banner=False, njobs=-1, chunksize=1)
        self.assertEqual(values.shape, (len(MOLECULES), 118))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 118)

    def test_3D_descriptor_size(self):
        values = self.blu3d.calculate(self.molecules, show_banner=False)
        print(values.columns.tolist())
        self.assertEqual(values.shape, (len(MOLECULES), 174))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 174)

    def test_3D_descriptor_multithread(self):
        values = self.blu3d.calculate(self.molecules, show_banner=False, njobs=-1, chunksize=1)
        self.assertEqual(values.shape, (len(MOLECULES), 174))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 174)

    def test_get_details(self):
        details = self.blu.get_details()
        self.assertEqual(details.shape, (174, 4))
        self.assertListEqual(details.columns.tolist(), ['Name', 'Description', 'Type', 'Dimensions'])

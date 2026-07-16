"""Tests for parallel dispatch: chunk sizing, core-count validation, JVM pinning."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from BlueDesc_pywrapper import BlueDesc
from BlueDesc_pywrapper.bluedesc_pywrapper import _make_chunks
from tests.constants import DIVERSE_SUBSET_SMALL

# Some CI runners (e.g. macOS) expose fewer than 4 cores; cap to what's actually available
# so calculate()'s njobs > cpu_count guard doesn't reject the request.
NJOBS = max(1, min(4, os.cpu_count() or 1))


class TestMakeChunks(unittest.TestCase):
    """Unit tests for the pure chunk-splitting helper (no Java involved)."""

    def test_no_molecules(self):
        self.assertEqual(_make_chunks([], njobs=4, chunksize=None), [])

    def test_auto_split_uses_all_workers_when_enough_molecules(self):
        mols = list(range(7))
        chunks = _make_chunks(mols, njobs=4, chunksize=None)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(sorted(len(c) for c in chunks), [1, 2, 2, 2])
        self.assertEqual(sum(chunks, []), mols)

    def test_auto_split_fewer_molecules_than_workers(self):
        mols = list(range(3))
        chunks = _make_chunks(mols, njobs=8, chunksize=None)
        # Cannot use more workers than there are molecules: exactly len(mols) chunks.
        self.assertEqual(len(chunks), 3)
        self.assertEqual(sorted(len(c) for c in chunks), [1, 1, 1])

    def test_explicit_chunksize_is_authoritative(self):
        mols = list(range(7))
        chunks = _make_chunks(mols, njobs=4, chunksize=1000)
        # A too-large explicit chunksize still yields a single chunk (old batching behavior).
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], mols)

    def test_explicit_small_chunksize(self):
        mols = list(range(7))
        chunks = _make_chunks(mols, njobs=4, chunksize=3)
        self.assertEqual([len(c) for c in chunks], [3, 3, 1])


class TestNjobsValidation(unittest.TestCase):
    """Tests for njobs bounds checking, without invoking Java."""

    def test_njobs_zero_raises(self):
        blu = BlueDesc()
        with self.assertRaises(ValueError):
            blu.calculate(DIVERSE_SUBSET_SMALL[:2], njobs=0, show_banner=False)

    def test_njobs_negative_raises(self):
        blu = BlueDesc()
        with self.assertRaises(ValueError):
            blu.calculate(DIVERSE_SUBSET_SMALL[:2], njobs=-1, show_banner=False)

    def test_njobs_exceeds_cpu_count_raises(self):
        blu = BlueDesc()
        with patch("os.cpu_count", return_value=2):
            with self.assertRaises(ValueError):
                blu.calculate(DIVERSE_SUBSET_SMALL[:2], njobs=3, show_banner=False)


class TestJvmCommandPinning(unittest.TestCase):
    """Tests that every Java invocation is pinned to a single core."""

    def test_prepare_command_pins_processor_count(self):
        blu = BlueDesc()
        command = blu._prepare_command(DIVERSE_SUBSET_SMALL[:2])
        self.addCleanup(os.remove, blu._tmp_sd)
        self.assertIsInstance(command, list)
        self.assertIn("-XX:ActiveProcessorCount=1", command)


class TestParallelMatchesSingleProcess(unittest.TestCase):
    """Ensures the njobs>1 code path is exercised and produces the same results as njobs=1."""

    def test_parallel_matches_single_process(self):
        mols = DIVERSE_SUBSET_SMALL[:12]
        single = BlueDesc().calculate(mols, show_banner=False, njobs=1)
        parallel = BlueDesc().calculate(mols, show_banner=False, njobs=NJOBS, chunksize=None)
        self.assertEqual(single.shape, parallel.shape)
        self.assertEqual(list(single.dtypes), list(parallel.dtypes))
        for col in single.columns:
            a = single[col].astype("float64").to_numpy()
            b = parallel[col].astype("float64").to_numpy()
            # rtol/atol rather than exact equality: BlueDesc's own JVM-side floating-point
            # summation is not perfectly reproducible across different SD-file batch sizes
            # (observed ~1e-6 relative drift on GRAVH-* between a 12-molecule single-process
            # batch and 3-molecule parallel chunks) -- not something this wrapper controls.
            self.assertTrue(np.allclose(a, b, rtol=1e-3, atol=1e-2, equal_nan=True), msg=col)

    def test_parallel_uses_all_requested_workers(self):
        # With 12 molecules, _make_chunks must produce NJOBS chunks: verified directly
        # (avoids depending on OS-level process introspection, which is flaky in CI).
        mols = DIVERSE_SUBSET_SMALL[:12]
        chunks = _make_chunks(mols, njobs=NJOBS, chunksize=None)
        self.assertEqual(len(chunks), NJOBS)

    def test_no_molecules_returns_empty_dataframe(self):
        blu = BlueDesc()
        self.assertTrue(blu.calculate([], show_banner=False, njobs=1).empty)
        self.assertTrue(blu.calculate([], show_banner=False, njobs=NJOBS).empty)


if __name__ == "__main__":
    unittest.main()

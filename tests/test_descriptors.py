"""Tests for BlueDesc molecular descriptors."""

import os
import subprocess
import unittest
import warnings
from unittest.mock import patch

from rdkit import Chem
from rdkit.Chem import AllChem

from BlueDesc_pywrapper import BlueDesc
from tests.constants import MOLECULES


class TestDescriptors(unittest.TestCase):
    """Tests for BlueDesc_pywrapper molecular descriptors."""

    def setUp(self) -> None:
        """Create the molecular descriptor calculators."""
        self.blu = BlueDesc()
        self.blu3d = BlueDesc(ignore_3D=False)
        self.molecules = list(MOLECULES.values())

    def test_2D_descriptor_size(self):
        values = self.blu.calculate(self.molecules, show_banner=False)
        self.assertEqual(values.shape, (len(MOLECULES), 118))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 118)

    def test_2D_descriptor_multithread(self):
        values = self.blu.calculate(self.molecules, show_banner=False, njobs=1, chunksize=1)
        self.assertEqual(values.shape, (len(MOLECULES), 118))
        self.assertFalse(values.isna().any().any())
        self.assertEqual(len(values.columns.unique().tolist()), 118)

    def test_3D_descriptor_size(self):
        values = self.blu3d.calculate(self.molecules, show_banner=False)
        self.assertEqual(values.shape, (len(MOLECULES), 174))
        # BlueDesc itself never populates these four 3D "unity"-weighted graph descriptors --
        # verified empirically: NaN for every molecule in a 73-molecule diverse sample, not
        # just an artifact of this small test set. Every other column is expected to be dense.
        always_nan_upstream = ["Wgamma1.unity", "Wgamma2.unity", "Wgamma3.unity", "WG.unity"]
        self.assertFalse(values.drop(columns=always_nan_upstream).isna().any().any())
        self.assertTrue(values[always_nan_upstream].isna().all().all())
        self.assertEqual(len(values.columns.unique().tolist()), 174)

    def test_known_reference_values(self):
        """MolecularWeight for erlotinib is a well-known reference value (393.44 g/mol);
        this pins both the V2000/V3000 SD-format choice and the int/float dtype fix, since
        both previously silently zeroed out this descriptor.
        """
        values = self.blu.calculate([MOLECULES["erlotinib"]], show_banner=False)
        self.assertAlmostEqual(float(values["MolecularWeight"].iloc[0]), 393.443, places=2)

    def test_get_details(self):
        details = self.blu.get_details()
        self.assertEqual(details.shape, (174, 4))
        self.assertListEqual(details.columns.tolist(), ["Name", "Description", "Type", "Dimensions"])

    def test_get_details_single_descriptor(self):
        details = self.blu.get_details("MolecularWeight")
        self.assertEqual(details.shape, (1, 4))
        self.assertEqual(details.Name.iloc[0], "MolecularWeight")

    def test_get_details_invalid_descriptor_raises(self):
        with self.assertRaises(ValueError):
            self.blu.get_details("NotADescriptor")

    def test_show_banner_logs(self):
        with self.assertLogs("BlueDesc_pywrapper.bluedesc_pywrapper", level="INFO"):
            self.blu.calculate([MOLECULES["erlotinib"]], show_banner=True)


class TestBlueDescValidation(unittest.TestCase):
    """Tests for input validation and error paths."""

    def test_missing_jar_raises_oserror(self):
        with patch.object(BlueDesc, "_jarfile", "/nonexistent/path/to.jar"):
            with self.assertRaises(OSError):
                BlueDesc()

    def test_atom_count_over_limit_raises(self):
        blu = BlueDesc()
        big_mol = Chem.AddHs(Chem.MolFromSmiles("C" * 1000))
        with self.assertRaises(ValueError):
            blu.calculate([big_mol], show_banner=False)

    def test_3d_descriptors_without_conformer_raises(self):
        blu3d = BlueDesc(ignore_3D=False)
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        with self.assertRaises(ValueError):
            blu3d.calculate([mol], show_banner=False)

    def test_missing_hydrogens_warns(self):
        blu = BlueDesc()
        mol = Chem.MolFromSmiles("n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C")
        AllChem.EmbedMolecule(mol, randomSeed=1)
        with self.assertWarns(UserWarning):
            blu.calculate([mol], show_banner=False)

    def test_missing_hydrogens_warning_is_never_swallowed_under_warnings_as_errors(self):
        """When warnings are configured to raise (e.g. pytest's -W error, or a caller's own
        warnings.simplefilter("error")), the missing-hydrogens UserWarning must propagate
        as an actual exception, not be silently absorbed by the surrounding try/except in
        _prepare_command (which only handled ValueError before this test was added) and
        without leaking the temporary SD file it was writing to.
        """
        blu = BlueDesc()
        mol = Chem.MolFromSmiles("n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C")
        AllChem.EmbedMolecule(mol, randomSeed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                blu.calculate([mol], show_banner=False)
        # No leftover self._tmp_sd attribute pointing at a file that still exists on disk.
        self.assertFalse(os.path.isfile(blu._tmp_sd))

    def test_arff_parsing_does_not_swallow_warnings_as_errors(self):
        """_run_command's except clause around arff.loadarff must only catch the specific
        parsing-failure types it is meant to translate into a clearer RuntimeError, never a
        bare Exception -- otherwise a warning promoted to an exception during parsing (e.g.
        under -W error, or a caller's own warnings.simplefilter("error")) would be silently
        absorbed and reported as an unrelated, misleading "no usable output" RuntimeError.
        """
        blu = BlueDesc()
        command = blu._prepare_command([MOLECULES["erlotinib"]])

        def cleanup():
            for path in (blu._tmp_sd, f"{blu._tmp_sd}.oddescriptors.arff", f"{blu._tmp_sd}.lp.oddescriptors.att"):
                if os.path.isfile(path):
                    os.remove(path)

        self.addCleanup(cleanup)
        with patch("BlueDesc_pywrapper.bluedesc_pywrapper.arff.loadarff",
                   side_effect=UserWarning("simulated warning promoted to an exception")):
            with self.assertRaises(UserWarning):
                blu._run_command(command)

    def test_flat_coordinates_raise_informative_runtimeerror(self):
        """BlueDesc/JOELib2 requires an actual 3D conformer, even for 1D/2D descriptors.

        Molecules that only carry flat/2D coordinates make every molecule get skipped
        internally, which used to surface as a bare scipy StopIteration; this should now
        raise a clear, actionable RuntimeError instead.
        """
        blu = BlueDesc()
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.Compute2DCoords(mol)
        with self.assertRaises(RuntimeError) as ctx:
            blu.calculate([mol], show_banner=False)
        self.assertIn("3D conformer", str(ctx.exception))

    def test_skipped_molecule_yields_aligned_nan_row(self):
        blu = BlueDesc()
        mols = [MOLECULES["erlotinib"], None, MOLECULES["lomitapide"]]
        values = blu.calculate(mols, show_banner=False)
        self.assertEqual(values.shape[0], 3)
        self.assertTrue(values.iloc[1].isna().all())
        self.assertFalse(values.iloc[0].isna().any())
        self.assertFalse(values.iloc[2].isna().any())

    def test_all_molecules_none_returns_all_nan_without_invoking_java(self):
        """Every molecule unparseable at the RDKit level: nothing gets written to the SD
        file, so BlueDesc must not be invoked on an empty input (which produces a
        header-only ARFF file indistinguishable from a genuine failure).
        """
        blu = BlueDesc()
        values = blu.calculate([None, None, None], show_banner=False)
        self.assertEqual(values.shape, (3, 118))
        self.assertTrue(values.isna().all().all())

    def test_subprocess_failure_surfaces_stderr(self):
        blu = BlueDesc()
        command = blu._prepare_command([MOLECULES["erlotinib"]])
        # Corrupt the jar argument so the JVM fails and stderr is captured.
        command[command.index("-jar") + 1] = "/nonexistent/bad.jar"
        with self.assertRaises(RuntimeError) as ctx:
            blu._run_command(command)
        self.assertTrue(str(ctx.exception))

    def test_cdk_internal_skip_is_reinserted_as_nan(self):
        """Exercise the "Sequence of skipped instances" realignment branch of _run_command.

        A real molecule that CDK skips internally is hard to obtain deterministically (it
        did not occur once across ~400 real, well-formed molecules), so this patches
        subprocess.run to simulate BlueDesc reporting one internally-skipped molecule out
        of three, matching the exact stdout format BlueDesc emits.
        """
        blu = BlueDesc()
        mols = [MOLECULES["erlotinib"], MOLECULES["midecamycin"], MOLECULES["lomitapide"]]
        command = blu._prepare_command(mols)
        real_out = f"{blu._tmp_sd}.oddescriptors.arff"
        real_process = subprocess.run(command, capture_output=True, text=True)
        # Rewrite the real output to drop the middle row and report it as internally skipped,
        # mirroring what BlueDesc does when it cannot compute descriptors for one molecule.
        with open(real_out) as fh:
            lines = fh.readlines()
        data_start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("@DATA"))
        header_lines = lines[:data_start + 1]
        data_lines = [line for line in lines[data_start + 1:] if line.strip() and not line.startswith("%")]
        self.assertEqual(len(data_lines), 3)
        truncated_lines = header_lines + [data_lines[0], data_lines[2]]
        with open(real_out, "w") as fh:
            fh.writelines(truncated_lines)
        fake_stdout = real_process.stdout + "\n\nSequence of skipped instances:\n2\n"

        class _FakeProcess:
            returncode = 0
            stdout = fake_stdout
            stderr = ""

        with patch("subprocess.run", return_value=_FakeProcess()):
            values = blu._run_command(command)
        blu._cleanup()
        self.assertEqual(values.shape[0], 3)
        self.assertTrue(values.iloc[1].isna().all())
        self.assertFalse(values.iloc[0].isna().any())
        self.assertFalse(values.iloc[2].isna().any())


if __name__ == "__main__":
    unittest.main()

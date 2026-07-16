"""Python wrapper for BlueDesc descriptors"""

import json
import logging
import multiprocessing
import os
import subprocess
import warnings
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import more_itertools
import numpy as np
import pandas as pd
from BlueDesc import BLUEDESC_EXEC_PATH
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from scipy.io import arff

from .utils import install_java, mktempfile, needsHs

logger = logging.getLogger(__name__)


def _make_chunks(mols: list, njobs: int, chunksize: int | None) -> list[list]:
    """Split molecules into chunks to be dispatched to worker processes.

    If chunksize is None, mols are split into exactly min(njobs, len(mols))
    balanced chunks (sizes differ by at most one) so every requested worker
    is guaranteed to receive work. If chunksize is given, it is authoritative
    and mols are batched into fixed-size chunks (may leave some workers idle).

    :param mols: molecules (or any list) to split into chunks
    :param njobs: number of concurrent workers that will consume the chunks
    :param chunksize: fixed size of each chunk, or None to auto-balance across njobs
    :return: a list of chunks
    """
    if not mols:
        return []
    if chunksize is None:
        n_chunks = min(njobs, len(mols))
        base, remainder = divmod(len(mols), n_chunks)
        chunks = []
        start = 0
        for i in range(n_chunks):
            size = base + (1 if i < remainder else 0)
            chunks.append(mols[start:start + size])
            start += size
        return chunks
    return [list(chunk) for chunk in more_itertools.batched(mols, chunksize)]


class BlueDesc:
    """Wrapper to obtain molecular descriptors from BlueDesc."""

    # Extra safety net for JRE installation; the real guarantee is that calculate() installs
    # the JRE once, before any worker starts.
    lock = multiprocessing.RLock()
    # Path to the JAR file
    _jarfile = BLUEDESC_EXEC_PATH
    # Path to the descriptor dtypes, cast onto results in _calculate
    _dtypes_file = os.path.abspath(os.path.join(__file__, os.pardir, 'dtypes.json'))

    def __init__(self, ignore_3D: bool = True):
        """Instantiate a wrapper to calculate BlueDesc molecular descriptors.

        :param ignore_3D: whether to skip the calculation of 3D molecular descriptors
        """
        self.include_3D = not ignore_3D
        # Ensure the jar file exists
        if not os.path.isfile(self._jarfile):
            raise OSError('The required BlueDesc JAR file is not present. Reinstall BlueDesc.')

    def calculate(self, mols: Iterable[Chem.Mol], show_banner: bool = True, njobs: int = 1,
                  chunksize: int | None = None) -> pd.DataFrame:
        """Calculate molecular descriptors.

        :param mols: RDKit molecules for which descriptors should be calculated
        :param show_banner: If True, show notice on descriptor usage
        :param njobs: number of concurrent processes; must not exceed the number of available
            CPU cores. Parallelism comes from spawning njobs OS processes, each running its own
            single-core-pinned JVM (-XX:ActiveProcessorCount=1), so JVMs do not oversubscribe
            the host by each auto-sizing GC/JIT thread pools to the full core count.
        :param chunksize: number of molecules processed per worker process; if None (default)
            molecules are auto-balanced across njobs workers so every worker gets work; ignored
            if njobs is 1.
        :return: a pandas DataFrame containing all BlueDesc descriptor values
        """
        if njobs < 1:
            raise ValueError('njobs must be a strictly positive integer.')
        cpu_count = os.cpu_count() or 1
        if njobs > cpu_count:
            raise ValueError(f'njobs ({njobs}) exceeds the number of available CPU cores ({cpu_count}).')
        if show_banner:
            self._show_banner()
        mols = list(mols)
        if not mols:
            return pd.DataFrame()
        # Parallelize should need be
        if njobs > 1:
            # Ensure the JRE is installed once in the parent process before workers start,
            # so concurrently-spawned workers all find it already in place.
            with self.lock:
                install_java()
            chunks = _make_chunks(mols, njobs, chunksize)
            with ProcessPoolExecutor(max_workers=njobs) as worker:
                futures = [worker.submit(self._multiproc_calculate, chunk) for chunk in chunks]
                results = [future.result() for future in futures]
            return pd.concat(results).reset_index(drop=True)
        # Single process
        return self._calculate(mols)

    def _show_banner(self):
        """Log info message for citing."""
        logger.info("""BlueDesc is a simple command-line tool converts an MDL SD file
into ARFF and LIBSVM format for machine learning and data mining purposes using
CDK and JOELib2. It computes 174 descriptors taken from both JOELib2 and the CDK.

###################################

Unfortunately there is no scientific article associated to BlueDesc, and the
following link is dead:
http://www.ra.cs.uni-tuebingen.de/software/bluedesc/welcome_e.html.

###################################
""")

    def _prepare_command(self, mols: list[Chem.Mol]) -> list[str]:
        """Create the BlueDesc command to be run to obtain molecular descriptors.

        :param mols: molecules to obtain molecular descriptors of
        :return: the command, as an argument list, to run.
        """
        # 1) Ensure JRE is accessible
        with self.lock:
            self._java_path = install_java()
        # 2) Create temp SD file
        self._tmp_sd = mktempfile('molecules_v2k.sd')
        self._skipped = []
        self.n = 0
        try:
            with BlockLogs():
                writer = Chem.SDWriter(self._tmp_sd)
                # Force V2000: BlueDesc/JOELib2 silently mis-parses the V3000 atom block --
                # elemental-composition descriptors (NumberOfC, NumberOfN, MolecularWeight,
                # LogP, PolarSurfaceArea, ...) come back as 0 under V3000, verified against
                # known reference values (e.g. erlotinib MolecularWeight == 393.443 under
                # V2000 vs. 0.0 under V3000). CDK genuinely cannot process V3000 here.
                writer.SetForceV3000(False)
                for i, mol in enumerate(mols):
                    if mol is not None and isinstance(mol, Chem.Mol):
                        if mol.GetNumAtoms() > 999:
                            raise ValueError('Cannot calculate descriptors for molecules with more than 999 atoms.')
                        # Does molecule lack hydrogen atoms?
                        if needsHs(mol):
                            warnings.warn(
                                'Molecule lacks hydrogen atoms: this might affect the value of calculated descriptors',
                                category=UserWarning, stacklevel=2,
                            )
                        # Are 3D descriptors requested?
                        if self.include_3D:
                            confs = list(mol.GetConformers())
                            if not (len(confs) > 0 and confs[-1].Is3D()):
                                raise ValueError('Cannot calculate the 3D descriptors of a conformer-less molecule')
                        writer.write(mol)
                    else:
                        self._skipped.append(i)
                    self.n += 1
                writer.close()
        except Exception:
            # Free resources on any failure -- not just the ValueErrors raised above, but
            # also e.g. the warnings.warn() call above being promoted to an exception by
            # -W error -- so the temp SD file is never leaked on an error path. A bare
            # `raise` re-raises whatever was caught completely unchanged (type, message,
            # traceback), so nothing raised here -- including a warning -- is ever masked.
            writer.close()
            os.remove(self._tmp_sd)
            raise
        # 3) Create command
        command = [
            self._java_path,
            '-Djava.awt.headless=true',
            # Pin each JVM to a single core: parallelism comes from spawning njobs OS
            # processes, not from letting each JVM auto-size GC/JIT threads to the host's
            # full core count (which would oversubscribe cores beyond what njobs requested).
            '-XX:ActiveProcessorCount=1',
            '-jar', self._jarfile,
            '-f', self._tmp_sd,
            '-l', '?',
        ]
        return command

    def _cleanup(self) -> None:
        """Cleanup resources used for calculation."""
        # Remove temporary files
        os.remove(self._tmp_sd)
        os.remove(self._out)
        att_file = f'{self._tmp_sd}.lp.oddescriptors.att'
        if os.path.isfile(att_file):
            os.remove(att_file)

    def _run_command(self, command: list[str]) -> pd.DataFrame:
        """Run the BlueDesc command.

        :param command: the command, as an argument list, to be run
        :return: a pandas DataFrame containing raw descriptor values, still aligned to the
            molecules actually written to the SD file (i.e. excluding self._skipped entries)
        """
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            # No output file was produced: only the SD input needs removing (self._out is not
            # yet set, so the full _cleanup(), which also removes self._out, cannot run here).
            os.remove(self._tmp_sd)
            raise RuntimeError(
                f'BlueDesc did not succeed to run properly (exit code {process.returncode}):\n{process.stderr}'
            )
        self._out = f'{self._tmp_sd}.oddescriptors.arff'
        try:
            values = arff.loadarff(self._out)
        except (arff.ArffError, ValueError, NotImplementedError, StopIteration) as e:
            # If every molecule was skipped internally (see the "Sequence of skipped
            # instances" handling below), BlueDesc writes a header-only, truncated ARFF file
            # that scipy cannot parse. This is known to happen for molecules that only carry
            # 2D/flat coordinates (e.g. via Compute2DCoords): unlike PaDEL, BlueDesc/JOELib2
            # requires an actual (non-degenerate) 3D conformer to compute descriptors at all,
            # even when only 1D/2D descriptors (ignore_3D=True, the default) are requested.
            self._cleanup()
            raise RuntimeError(
                'BlueDesc produced no usable output -- every molecule was likely skipped '
                'internally. This commonly happens when molecules only carry flat/2D '
                'coordinates: BlueDesc requires an embedded 3D conformer (e.g. via '
                'rdkit.Chem.AllChem.EmbedMolecule) to compute descriptors, even the 1D/2D '
                'ones.'
            ) from e
        values = (pd.DataFrame(values[0])
                  .drop("?", axis=1)
                  .rename(columns=lambda x: x.replace('joelib2.feature.types.count.', ''))
                  .rename(columns=lambda x: x.replace('joelib2.feature.types.', ''))
                  )
        expected_rows = self.n - len(self._skipped)
        if values.shape[0] != expected_rows:
            # BlueDesc itself skipped some molecules internally (e.g. it could not compute
            # descriptors for them); reinsert NaN rows at the positions it reports so the
            # output stays aligned with the molecules written to the SD file.
            values_ = pd.DataFrame(np.full((expected_rows, values.shape[1]), np.nan), columns=values.columns)
            start = process.stdout.find('Sequence of skipped instances:')
            skipped = pd.Series(process.stdout[start + 30:].split()).astype(int) - 1
            skipped = set(skipped.tolist())
            for i in range(expected_rows):
                if i not in skipped:
                    values_.iloc[i, :] = values.iloc[0, :]
                    values = values.iloc[1:, :]
            values = values_
        # If only 2D, remove 3D descriptors
        if not self.include_3D:
            descs_3D = self.get_details()
            descs_3D = descs_3D[descs_3D.Dimensions == '3D']
            values = values.drop(columns=descs_3D.Name.tolist())
        return values

    def _calculate(self, mols: list[Chem.Mol]) -> pd.DataFrame:
        """Calculate BlueDesc molecular descriptors on one process.

        :param mols: RDKit molecules for which BlueDesc descriptors should be calculated.
        :return: a pandas DataFrame containing BlueDesc descriptor values
        """
        # Prepare inputs
        command = self._prepare_command(mols)
        if len(self._skipped) == len(mols):
            # Every molecule was unparseable at the RDKit level: nothing was written to the SD
            # file, so there is nothing for BlueDesc to run on. Short-circuit here instead of
            # invoking BlueDesc on an empty input, which produces a header-only ARFF file that
            # _run_command cannot meaningfully distinguish from a genuine failure.
            os.remove(self._tmp_sd)
            details = self.get_details()
            if not self.include_3D:
                details = details[details.Dimensions != '3D']
            results = pd.DataFrame(np.nan, index=range(len(mols)), columns=details.Name.tolist())
            return self._cast_dtypes(results)
        # Run command and obtain results
        results = self._run_command(command)
        # Cleanup
        self._cleanup()
        # Insert lines of skipped molecules
        if len(self._skipped):
            results = pd.DataFrame(np.insert(results.values, self._skipped,
                                              values=[np.nan] * len(results.columns),
                                              axis=0),
                                    columns=results.columns)
        return self._cast_dtypes(results)

    def _cast_dtypes(self, results: pd.DataFrame) -> pd.DataFrame:
        """Cast descriptor columns to their proper dtypes.

        Nullable pandas dtypes ("Int64"/"float64"), rather than plain numpy "int", are
        required because skipped molecules leave NaN in otherwise-integer columns (e.g.
        NumberOfAtoms), which numpy int dtypes cannot hold.

        :param results: raw descriptor values, as parsed from BlueDesc's ARFF output (or an
            all-NaN placeholder when every molecule was skipped)
        :return: results with every column cast to its declared "Int64"/"float64" dtype
        """
        numpy_to_nullable = {'int': 'Int64', 'float': 'float64'}
        with open(self._dtypes_file) as fh:
            raw_dtypes = json.load(fh)
        dtypes = {name: numpy_to_nullable[dtype] for name, dtype in raw_dtypes.items() if name in results.columns}
        return results.apply(pd.to_numeric, errors='coerce').astype(dtypes)

    def _multiproc_calculate(self, mols: list[Chem.Mol]) -> pd.DataFrame:
        """Calculate BlueDesc descriptors in a worker process.

        :param mols: RDKit molecules for which BlueDesc descriptors should be calculated
        :return: a pandas DataFrame containing all BlueDesc descriptor values
        """
        # Copy self instance to make thread safe
        bluedesc = deepcopy(self)
        # Run copy
        result = bluedesc.calculate(mols, show_banner=False, njobs=1)
        return result

    @staticmethod
    def get_details(desc_name: str | None = None) -> pd.DataFrame:
        """Obtain details about either one or all descriptors.

        :param desc_name: the name of the descriptor to obtain details about (default: None).
            If None, returns details about all descriptors.
        :return: a pandas DataFrame detailing the name, description, type and dimensionality
            of the requested descriptor(s)
        """
        details = pd.read_json(os.path.abspath(os.path.join(__file__, os.pardir, 'descs.json')), orient='index')
        if desc_name is not None:
            if desc_name not in details.Name.tolist():
                raise ValueError(f'descriptor name {desc_name} is not available')
            details = details[details.Name == desc_name]
        return details

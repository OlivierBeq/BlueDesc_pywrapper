# -*- coding: utf-8

"""Python wrapper for BlueDesc descriptors"""

from __future__ import annotations

import json
import more_itertools
import multiprocessing
import numpy as np
import os
import pandas as pd
import subprocess
import warnings
from copy import deepcopy
from typing import Iterable, List, Optional

from BlueDesc import BLUEDESC_EXEC_PATH
from bounded_pool_executor import BoundedProcessPoolExecutor
from rdkit import Chem
from rdkit.rdBase import BlockLogs
from scipy.io import arff

from .utils import install_java, mktempfile, needsHs


class BlueDesc:
    """Wrapper to obtain molecular descriptor from BlueDesc."""

    lock = multiprocessing.RLock()  # Ensure installation of JRE is thread safe
    # Path to the JAR file
    _jarfile = BLUEDESC_EXEC_PATH

    def __init__(self, ignore_3D: bool = True):
        """Instantiate a wrapper to calculate BlueDesc molecular descriptors.

        :param ignore_3D: whether to include 3D molecular descriptors
        """
        self.include_3D = not ignore_3D
        # Ensure the jar file exists
        if not os.path.isfile(self._jarfile):
            raise IOError('The required BlueDesc JAR file is not present. Reinstall BlueDesc.')

    def calculate(self, mols: Iterable[Chem.Mol], show_banner: bool = True, njobs: int = 1,
                  chunksize: Optional[int] = 1000) -> pd.DataFrame:
        """Calculate molecular descriptors.

        :param mols: RDKit molecules for which descriptors should be calculated
        :param show_banner: If True, show notice on descriptor usage
        :param njobs: number of concurrent processes
        :param chunksize: number of molecules to be processed by a process; ignored if njobs is 1
        :return: a pandas DataFrame containing all BlueDesc descriptor values
        """
        if show_banner:
            self._show_banner()
        if njobs < 0:
            njobs = os.cpu_count() - njobs + 1
        # Parallelize should need be
        if njobs != 1:
            if njobs < 0:
                njobs += multiprocessing.cpu_count() + 1
            with BoundedProcessPoolExecutor(max_workers=njobs) as worker:
                futures = [worker.submit(self._multiproc_calculate, list(chunk))
                           for chunk in more_itertools.batched(mols, chunksize)
                           ]
            data = (pd.concat([future.result()
                               for future in futures]
                              )
                    .reset_index(drop=True))
        # Single process
        else:
            data = self._calculate(list(mols))
        return data


    def _show_banner(self):
        """Print info message for citing."""
        print("""BlueDesc is a simple command-line tool converts an MDL SD file
into ARFF and LIBSVM format for machine learning and data mining purposes using
CDK and JOELib2. It computes 174 descriptors taken from both JOELib2 and the CDK.

###################################

Unfortunately there is no scientific article associated to BlueDesc, and the
following link is dead:
http://www.ra.cs.uni-tuebingen.de/software/bluedesc/welcome_e.html.

###################################

""")

    def _prepare_command(self, mols: List[Chem.Mol]) -> str:
        """Create the BlueDesc command to be run to obtain molecular descriptors.

        :param mols: molecules to obtained molecular descriptors of
        :return: The command to run.
        """
        # 1) Ensure JRE is accessible
        with self.lock:
            self._java_path = install_java()
        # 2) Create temp SD v2k file
        self._tmp_sd = mktempfile('molecules_v3k.sd')
        self._skipped = []
        self.n = 0
        try:
            block = BlockLogs()
            writer = Chem.SDWriter(self._tmp_sd)
            # Ensure V2000 as CDK cannot properly process v3000
            writer.SetForceV3000(True)
            for i, mol in enumerate(mols):
                if mol is not None and isinstance(mol, Chem.Mol):
                    if mol.GetNumAtoms() > 999:
                        raise ValueError('Cannot calculate descriptors for molecules with more than 999 atoms.')
                    # Does molecule lack hydrogen atoms?
                    if needsHs(mol):
                        warnings.warn('Molecule lacks hydrogen atoms: this might affect the value of calculated descriptors')
                    # Are molecules 3D
                    if self.include_3D:
                        confs = list(mol.GetConformers())
                        if not (len(confs) > 0 and confs[-1].Is3D()):
                            raise ValueError('Cannot calculate the 3D descriptors of a conformer-less molecule')
                    writer.write(mol)
                else:
                    self._skipped.append(i)
                self.n += 1
            writer.close()
            del block
        except ValueError as e:
            # Free resources and raise error
            writer.close()
            del block
            os.remove(self._tmp_sd)
            raise e from None
        # 3) Create command
        java_path = install_java()
        command = f"{java_path} -jar {self._jarfile} -f {self._tmp_sd} -l ?"
        return command

    def _cleanup(self) -> None:
        """Cleanup resources used for calculation."""
        # Remove temporary files
        os.remove(self._tmp_sd)
        os.remove(self._out)
        os.remove(f'{self._tmp_sd}.lp.oddescriptors.att')

    def _run_command(self, command: str) -> pd.DataFrame:
        """Run the BlueDesc command.

        :param command: The command to be run.
        """
        process = subprocess.run(command.split(), capture_output=True, text=True)
        if process.returncode == 0:
            self._out = f'{self._tmp_sd}.oddescriptors.arff'
            try:
                values = arff.loadarff(self._out)
                values = (pd.DataFrame(values[0])
                          .drop("?", axis=1)
                          .rename(columns=lambda x: x.replace('joelib2.feature.types.count.', ''))
                          .rename(columns=lambda x: x.replace('joelib2.feature.types.', ''))
                          )
                if values.shape[0] != (self.n - len(self._skipped)):
                    # Create an empty array
                    values_ = pd.DataFrame(np.zeros(((self.n - len(self._skipped)), values.shape[1])),
                                           columns=values.columns)
                    # Identify skipped molecules
                    start = process.stdout.find('Sequence of skipped instances:')
                    skipped = pd.Series(process.stdout[start + 30:].split()).astype(int) - 1
                    for i in range((self.n - len(self._skipped))):
                        if i in skipped.tolist():
                            values_.iloc[i, :] = np.NaN
                        else:
                            values_.iloc[i, :] = values.iloc[0, :]
                            values = values.iloc[1:, :]
                    values = values_
            except Exception as e:
                raise e
                details = self.get_details()
                values = np.zeros((self.n, details.shape[0]))
                with open(os.path.abspath(os.path.join(__file__, os.pardir, 'dtypes.json'))) as fh:
                    dtypes = json.load(fh)
                values = pd.DataFrame(values, columns=details.Name.tolist()).astype(dtypes)
            # If only 2D, remove 3D descriptors
            if not self.include_3D:
                # Get 3D descriptor names to remove
                descs_3D = self.get_details()
                descs_3D = descs_3D[descs_3D.Dimensions == '3D']
                values = values.drop(columns=descs_3D.Name.tolist())
        else:
            self._cleanup()
            raise RuntimeError('BlueDesc did not succeed to run properly.')
        return values

    def _calculate(self, mols: List[Chem.Mol]) -> pd.DataFrame:
        """Calculate BlueDesc molecular descriptors on one process.

        :param mols: RDKit molecules for which BlueDesc descriptors should be calculated.
        :return: a pandas DataFrame containing BlueDesc descriptor values
        """
        # Prepare inputs
        command = self._prepare_command(mols)
        # Run command and obtain results
        results = self._run_command(command)
        # Cleanup
        self._cleanup()
        # Insert lines of skipped molecules
        if len(self._skipped):
            results = (pd.DataFrame(np.insert(results.values, self._skipped,
                                              values=[np.NaN] * len(results.columns),
                                              axis=0),
                                    columns=results.columns)
                       )
        results = (results.apply(pd.to_numeric, errors='ignore', downcast='integer', axis=0)
                          .apply(pd.to_numeric, errors='ignore', downcast='float', axis=0)
                   )
        return results

    def _multiproc_calculate(self, mols: List[Chem.Mol], nbits: int = 1024) -> pd.DataFrame:
        """Calculate BlueDesc descriptors in thread-safe manner.

        :param mols: RDKit molecules for which BlueDesc descriptors should be calculated
        :return: a pandas DataFrame containing all BlueDesc desciptor values
        """
        # Copy self instance to make thread safe
        bluedesc = deepcopy(self)
        # Run copy
        result = bluedesc.calculate(mols, show_banner=False, njobs=1)
        return result

    @staticmethod
    def get_details(desc_name: Optional[str] = None):
        """Obtain details about either one or all descriptors.

        :param desc_name: the name of the descriptor to obtain details about (default: None).
        If None, returns details about all descriptors.
        """
        details = pd.read_json(os.path.abspath(os.path.join(__file__, os.pardir, 'descs.json')), orient='index')
        if desc_name is not None:
            if desc_name not in details.Name.tolist():
                raise ValueError(f'descriptor name {desc_name} is not available')
            details = details[details.Name == desc_name]
        return details

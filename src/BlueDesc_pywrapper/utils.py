"""Utility functions."""

import glob
import os
import sys
import tempfile

from filelock import FileLock
from jdk import _JRE_DIR
from jdk import install as _jre_install
from rdkit import Chem


def install_java(version: int = 11) -> str | None:
    """Install a Java Runtime Environment.

    Guarded by a cross-process file lock: ``jdk.install`` downloads to a fixed,
    version-derived temp file path and removes it once done, so concurrent callers
    (e.g. separate worker processes racing on a cold cache) can collide -- one
    process's download/extraction stepping on another's, or removing the shared
    temp file out from under it.

    :param version: major Java version to install
    :return: path to the ``java`` executable, or None if it could not be found
    """
    os.makedirs(_JRE_DIR, exist_ok=True)
    with FileLock(os.path.join(_JRE_DIR, f'.install-{version}.lock')):
        path = get_java_in_dir(_JRE_DIR, version)
        if path is None:
            # Could not find JRE, install it
            _jre_install(version, jre=True)
            path = get_java_in_dir(_JRE_DIR, version=version)
    return path


def get_java_in_dir(dir: str, version: int) -> str | None:
    """Recursively search the directory to find a JRE.

    :param dir: directory to search
    :param version: major Java version being searched for
    :return: absolute path to the ``java`` executable, or None if not found
    """
    paths = glob.glob(os.path.join(dir, '**', 'bin',
                                   'java.exe' if sys.platform == "win32" else 'java'
                                   ), recursive=True)
    path = [path for path in paths if f'jdk-{version}' in path]
    if len(path):
        return os.path.abspath(path[0])
    return None


def mktempdir(suffix: str | None = None) -> str:
    """Return the path to a writeable temporary directory.

    :param suffix: optional suffix to append to the directory name
    """
    return tempfile.mkdtemp(suffix=suffix)


def mktempfile(suffix: str | None = None) -> str:
    """Return the path to a writeable temporary file.

    :param suffix: optional suffix to append to the file name
    """
    file = tempfile.mkstemp(suffix=suffix)
    os.close(file[0])
    return file[1]


def needsHs(mol: Chem.Mol) -> bool:
    """Return if the molecule lacks hydrogen atoms or not.

    :param mol: RDKit Molecule
    :return: True if the molecule lacks hydrogens.
    """
    for atom in mol.GetAtoms():
        nHNbrs = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                nHNbrs += 1
        noNeighbors = False
        if atom.GetTotalNumHs(noNeighbors) > nHNbrs:
            return True
    return False

# 🧬 BlueDesc Python Wrapper

<!-- Badges -->
<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/BlueDesc-pywrapper.svg)](https://pypi.org/project/BlueDesc-pywrapper/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/BlueDesc-pywrapper.svg)](https://pypi.org/project/BlueDesc-pywrapper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/OlivierBeq/BlueDesc_pywrapper/actions/workflows/tests.yml/badge.svg)](https://github.com/OlivierBeq/BlueDesc_pywrapper/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<br>
</div>

A simple and reliable Python wrapper for calculating BlueDesc molecular descriptors. This library takes care of installing a matching Java runtime, dispatching molecules to the bundled BlueDesc executable, and collecting the results into a tidy pandas DataFrame — so you can stay in RDKit/pandas-land.

## ✨ Features

- 🧩 **174 descriptors** — 33 1D and 85 2D descriptors by default, plus 56 additional 3D descriptors on request, taken from both JOELib2 and the CDK.
- ☕ **Zero Java setup** — automatically downloads, caches and reuses a matching JRE on first use; nothing to install by hand.
- ⚡ **Parallel by design** — spread the work across multiple CPU cores with configurable `njobs`/`chunksize`, each worker running its own single-core-pinned JVM.
- 🧯 **Never silently misaligned** — molecules that fail to parse, or that BlueDesc itself cannot compute descriptors for, are skipped explicitly and reinserted as `NaN` rows, never dropped without a trace, regardless of `njobs`/`chunksize`.
- 📊 **pandas-native output** — results come back as a ready-to-use `DataFrame`, with descriptor columns cast to proper nullable integer/float dtypes.
- 🔍 **Rich metadata** — inspect each descriptor's name, description, category and dimensionality via `get_details()`.

## ✍️ Copyright and Citation Notice

Olivier J. M. Béquignon is **neither** the copyright holder of BlueDesc **nor** responsible for it. The work carried out here concerns solely the Python wrapper.

### Citing

Unfortunately there is no scientific article associated with BlueDesc, and the original project page (`http://www.ra.cs.uni-tuebingen.de/software/bluedesc/welcome_e.html`) is no longer online. If you use this wrapper in your research, please cite this software package:

> Béquignon, O. J. M. *BlueDesc_pywrapper: a Python wrapper for BlueDesc molecular descriptors.* https://github.com/OlivierBeq/BlueDesc_pywrapper

## 📦 Installation

```bash
pip install bluedesc-pywrapper
```

Or from source:

```bash
git clone https://github.com/OlivierBeq/BlueDesc_pywrapper.git
pip install ./BlueDesc_pywrapper
```

## 🛠️ Requirements

- Python 3.11+
- [RDKit](https://www.rdkit.org/docs/Install.html)

## 💡 Usage

### Getting started

> ⚠️ Unlike most CDK-based wrappers, BlueDesc/JOELib2 requires every molecule to carry an
> actual, non-degenerate **3D conformer** to compute descriptors at all — even its 1D/2D
> ones. Molecules with only flat/2D coordinates (e.g. from `Compute2DCoords`) or no
> coordinates at all are not supported: BlueDesc will silently fail to compute anything for
> them. Always embed a 3D conformer first, as in the example below.

```python
from BlueDesc_pywrapper import BlueDesc
from rdkit import Chem
from rdkit.Chem import AllChem

smiles_list = [
    # erlotinib
    "n1cnc(c2cc(c(cc12)OCCOC)OCCOC)Nc1cc(ccc1)C#C",
    # midecamycin
    "CCC(=O)O[C@@H]1CC(=O)O[C@@H](C/C=C/C=C/[C@@H]([C@@H](C[C@@H]([C@@H]([C@H]1OC)O[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)C)O[C@H]3C[C@@]([C@H]([C@@H](O3)C)OC(=O)CC)(C)O)N(C)C)O)CC=O)C)O)C",
    # selenofolate
    "C1=CC(=CC=C1C(=O)NC(CCC(=O)OCC[Se]C#N)C(=O)O)NCC2=CN=C3C(=N2)C(=O)NC(=N3)N",
]
mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
for mol in mols:
    AllChem.EmbedMolecule(mol)

bluedesc = BlueDesc()
print(bluedesc.calculate(mols))
```

The above calculates 118 molecular descriptors (33 1D and 85 2D).

> ⚠️ BlueDesc skips molecules it cannot parse, or that it fails to compute descriptors for
> internally — a warning is given when the latter happens. Skipped molecules always come
> back as an all-`NaN` row at their original position, so input and output stay aligned no
> matter how many molecules are skipped or how `njobs`/`chunksize` are set.

### 3D descriptors

The additional 56 three-dimensional (3D) descriptors can be computed like so:

```python
bluedesc = BlueDesc(ignore_3D=False)
print(bluedesc.calculate(mols))
```

> ⚠️ An exception is raised if a molecule lacks a 3D conformer when 3D descriptors are requested. <br/>
> ⚠️ Four of BlueDesc's own 3D descriptors (`Wgamma1.unity`, `Wgamma2.unity`, `Wgamma3.unity`,
> `WG.unity`) are never populated by the underlying tool itself — they always come back as
> `NaN`, independently of this wrapper.

### ⚡ Parallel processing

Speed things up by spreading molecules across several CPU cores. Each worker runs its own single-core-pinned JVM, so parallelism comes purely from the number of processes spawned — not from oversubscribing the host:

```python
bluedesc = BlueDesc()
print(bluedesc.calculate(mols, njobs=8))
```

By default, molecules are auto-balanced evenly across `njobs` workers (`chunksize=None`), which minimizes JVM startup overhead while keeping every worker busy — the fastest setting for most workloads. A fixed `chunksize` can be provided instead if finer control is needed.

### 🔍 Details about descriptors

Details about each descriptor can be obtained as follows:

```python
print(BlueDesc.get_details())

print(BlueDesc.get_details("MolecularWeight"))
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/OlivierBeq/BlueDesc_pywrapper/blob/master/LICENSE) file for details.

## 📚 API Documentation

```python
def calculate(mols, show_banner=True, njobs=1, chunksize=None):
```

Calculates BlueDesc molecular descriptors. Installs a matching JRE on first use if none is found.

#### Parameters

- ***mols  : Iterable[Chem.Mol]***
  RDKit molecule objects for which to obtain BlueDesc descriptors. Each molecule must carry
  an embedded 3D conformer (see the warning above).
- ***show_banner  : bool***
  Displays default notice about BlueDesc.
- ***njobs  : int***
  Number of concurrent processes used to calculate descriptors in parallel; must not exceed the
  number of available CPU cores. Each spawned Java process is pinned to a single core
  (`-XX:ActiveProcessorCount=1`), since parallelism comes from spawning `njobs` OS processes
  rather than from letting each JVM oversubscribe the host's full core count.
- ***chunksize  : int | None***
  Number of molecules processed per worker process. If `None` (default), molecules are
  auto-balanced across all `njobs` workers so every worker gets work. Ignored if `njobs` is 1.
- ***return_type  : pd.DataFrame***
  Pandas DataFrame containing BlueDesc descriptor values, one row per molecule.

________________

```python
BlueDesc(ignore_3D=True)
```

Wrapper to obtain molecular descriptors from BlueDesc.

#### Parameters

- ***ignore_3D  : bool***
  Whether to skip the calculation of the 56 3D descriptors. Default: `True`.

________________

```python
BlueDesc.get_details(desc_name=None)
```

Static method returning metadata about either one or all 174 descriptors.

#### Parameters

- ***desc_name  : str | None***
  Name of the descriptor to obtain details about. If `None` (default), returns details about
  all descriptors, as a DataFrame with `Name`, `Description`, `Type` and `Dimensions` columns.

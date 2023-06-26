[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Python wrapper for BlueDesc molecular descriptors

## Installation

From source:

    git clone https://github.com/OlivierBeq/bluedesc_pywrapper.git
    pip install ./bluedesc_pywrapper

with pip:

```bash
pip install BlueDesc-pywrapper
```

### Get started

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
  # cisplatin
  "N.N.Cl[Pt]Cl"
]
mols = [Chem.AddHs(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
for mol in mols:
    _ = AllChem.EmbedMolecule(mol)

bluedesc = BlueDesc()
print(bluedesc.calculate(mols))
```

The above calculates 174 molecular descriptors (33 1D, 85 2D and 56 3D).
:warning: Molecules are required to have conformers for descriptors to be calculated. 

```
## Documentation

```python
def calculate(mols, show_banner=True, njobs=1, chunksize=1000):
```

Default method to calculate BlueDesc fingerprints.

Parameters:

- ***mols  : Iterable[Chem.Mol]***  
  RDKit molecule objects for which to obtain BlueDesc descriptors.
- ***show_banner  : bool***  
  Displays default notice about BlueDesc.
- ***njobs  : int***  
  Maximum number of simultaneous processes.
- ***chunksize  : int***  
  Maximum number of molecules each process is charged of.

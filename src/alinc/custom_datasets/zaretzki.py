import ast
import os.path as osp
import logging
import torch

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, InMemoryDataset, download_url
from typing import List, Optional, Callable


logger = logging.getLogger(__name__)

# =============================================================================
# 1. OGB-Analogous Feature Extractors
#    (Re-implementation of https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py)
# =============================================================================

ALLOWABLE_FEATURES = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER', 'misc'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index."""
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """Converts rdkit atom into OGB-style feature vector (9 dimensions)."""
    atom_feature = [
        safe_index(ALLOWABLE_FEATURES['possible_atomic_num_list'], atom.GetAtomicNum() + 1),
        safe_index(ALLOWABLE_FEATURES['possible_chirality_list'], str(atom.GetChiralTag())),
        safe_index(ALLOWABLE_FEATURES['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(ALLOWABLE_FEATURES['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(ALLOWABLE_FEATURES['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(ALLOWABLE_FEATURES['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(ALLOWABLE_FEATURES['possible_hybridization_list'], str(atom.GetHybridization())),
        safe_index(ALLOWABLE_FEATURES['possible_is_aromatic_list'], atom.GetIsAromatic()),
        safe_index(ALLOWABLE_FEATURES['possible_is_in_ring_list'], atom.IsInRing()),
    ]
    return atom_feature

def bond_to_feature_vector(bond):
    """Converts rdkit bond into OGB-style feature vector (3 dimensions)."""
    bond_feature = [
        safe_index(ALLOWABLE_FEATURES['possible_bond_type_list'], str(bond.GetBondType())),
        safe_index(ALLOWABLE_FEATURES['possible_bond_stereo_list'], str(bond.GetStereo())),
        safe_index(ALLOWABLE_FEATURES['possible_is_conjugated_list'], bond.GetIsConjugated()),
    ]
    return bond_feature


def murcko_scaffold_smiles(mol):
    """Return a Murcko scaffold SMILES while tolerating RDKit STEREOANY bonds."""
    mol_for_scaffold = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol_for_scaffold)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol_for_scaffold)
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)

# =============================================================================
# 2. Zaretzki Dataset Class
#    Raw data source: https://github.com/molinfo-vienna/FAME.AL
# =============================================================================

class ZaretzkiDataset(InMemoryDataset):
    url = (
        "https://raw.githubusercontent.com/molinfo-vienna/FAME.AL/"
        "main/data/zaretzki_preprocessed.sdf"
    )

    def __init__(
            self, 
            root: str,
            split: str = "train",  # "train", "val", or "test"
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            force_reload: bool = False,
            seed: int = 42
        ) -> None:
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            transform (callable, optional): A function/transform that takes in an
                torch_geometric.data.Data object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in
                an torch_geometric.data.Data object and returns a transformed version.
            seed (int): Random seed for the train/val/test split.
        """

        if not split in ["train", "val", "test"]:
            raise KeyError(f"Invalid split {split}!")
        self.split = split
        self.seed = seed
        dataset_root = root + "/Zaretzki/"
        processed_paths = [
            osp.join(dataset_root, "processed", f"zaretzki_dataset_{split_name}.pt")
            for split_name in ["train", "val", "test"]
        ]
        if not force_reload and any(osp.exists(path) for path in processed_paths):
            force_reload = not all(osp.exists(path) for path in processed_paths)
            if force_reload:
                logger.warning("Incomplete Zaretzki processed files found; reprocessing.")

        super(ZaretzkiDataset, self).__init__(
            dataset_root, transform, pre_transform, pre_filter,
            force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return ["zaretzki_preprocessed.sdf"]
    
    @property
    def raw_paths(self) -> List[str]:
        return [osp.join(self.raw_dir, f) for f in self.raw_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"zaretzki_dataset_{self.split}.pt"]
    
    @property
    def processed_paths(self) -> List[str]:
        return [osp.join(self.processed_dir, f) for f in self.processed_file_names]

    def download(self):
        download_url(self.url, self.raw_dir, filename=self.raw_file_names[0])

    def process(self):
        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)
        data_list = []
        all_scaffolds = []
        
        for i, mol in enumerate(suppl):
            if mol is None:
                continue

            # --- STEP 1: PARSE 'soms' LIST STRING ---
            som_indices = set()
            if mol.HasProp('soms'):
                raw_val = mol.GetProp('soms') # e.g., "[15, 16]"
                try:
                    # ast.literal_eval safely converts "[15, 16]" -> [15, 16]
                    parsed_list = ast.literal_eval(raw_val)
                    
                    som_indices = {int(idx) for idx in parsed_list}
                except (ValueError, SyntaxError):
                    logger.warning("Error parsing soms for mol %s: %s", i, raw_val)

            # 2. Node Features (x) and labels (y)
            atom_features = []
            y_labels = []

            for idx, atom in enumerate(mol.GetAtoms()):
                # Feature Vector
                atom_features.append(atom_to_feature_vector(atom))
                
                # Check if this atom's 0-based index is in our SoM set
                if idx in som_indices:
                    y_labels.append(1)
                else:
                    y_labels.append(0)

            x = torch.tensor(atom_features, dtype=torch.long)
            y = torch.tensor(y_labels, dtype=torch.long)

            # 3. Edge Features (edge_attr) & Connectivity (edge_index)
            edge_indices = []
            edge_attrs = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                feat = bond_to_feature_vector(bond)
                
                # Add edges in both directions (undirected graph)
                edge_indices.append([i, j])
                edge_attrs.append(feat)
                edge_indices.append([j, i])
                edge_attrs.append(feat)

            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            else:
                # Handle molecules with no bonds (single atoms)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 3), dtype=torch.long)

            # Store SMILES for scaffold calculation
            smiles = Chem.MolToSmiles(mol)
            all_scaffolds.append(murcko_scaffold_smiles(mol))

            # Create Data Object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)

        # 3. Perform Scaffold Split (Bemis-Murcko)
        scaffolds = defaultdict(list)
        for i, scaffold in enumerate(all_scaffolds):
            scaffolds[scaffold].append(i)
        
        # Sort scaffolds by size (descending) - this is the standard OGB/MoleculeNet way
        scaffold_sets = [scaffolds[s] for s in sorted(scaffolds.keys(), key=lambda x: len(scaffolds[x]), reverse=True)]
        
        train_cutoff = int(0.8 * len(data_list))
        val_cutoff = int(0.9 * len(data_list))
        
        train_idx, val_idx, test_idx = [], [], []
        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= train_cutoff:
                train_idx.extend(scaffold_set)
            elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cutoff:
                val_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
        
        # Save the splits
        for split, indices in zip(["train", "val", "test"], [train_idx, val_idx, test_idx]):
            torch.save(
                self.collate([data_list[i] for i in indices]),
                osp.join(self.processed_dir, f"zaretzki_dataset_{split}.pt")
            )

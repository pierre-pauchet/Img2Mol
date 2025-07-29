from rdkit import Chem
import numpy as np
from qm9.bond_analyze import get_bond_order, geom_predictor
from . import dataset
import torch
from configs.datasets_config import get_dataset_info
import pickle
import os


def compute_qm9_smiles(dataset_name, remove_h):
    """
    Converts the QM9 dataset into SMILES strings.

    This function processes the QM9 dataset (or its second half) and converts
    molecular data into SMILES (Simplified Molecular Input Line Entry System) 
    representations. It supports optional removal of hydrogen atoms.

    Args:
        dataset_name (str): The name of the dataset to process. 
                            Expected values are 'qm9' or 'qm9_second_half'.
        remove_h (bool): A flag indicating whether to remove hydrogen atoms 
                         from the molecules.

    Returns:
        list: A list of SMILES strings representing the molecules in the dataset.

    Notes:
        - The function uses a static argument class to configure dataset loading.
        - The dataset is processed in batches, and progress is printed every 1000 molecules.
        - The function relies on external utilities such as `dataset.retrieve_dataloaders`, 
          `get_dataset_info`, `build_molecule`, and `mol2smiles`.
    """
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print("\tConverting QM9 dataset to SMILES ...")

    class StaticArgs:
        def __init__(self, dataset, remove_h):
            self.dataset = dataset
            self.batch_size = 1
            self.num_workers = 1
            self.filter_n_atoms = None
            self.datadir = 'qm9/temp'
            self.remove_h = remove_h
            self.include_charges = True
    args_dataset = StaticArgs(dataset_name, remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_dataset)
    dataset_info = get_dataset_info(args_dataset.dataset, args_dataset.remove_h)
    n_types = 4 if remove_h else 5
    mols_smiles = []
    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'][0].view(-1, 3).numpy()
        one_hot = data['one_hot'][0].view(-1, n_types).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        mol = build_molecule(torch.tensor(positions), torch.tensor(atom_type), dataset_info)
        mol = mol2smiles(mol)
        if mol is not None:
            mols_smiles.append(mol)
        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i)/len(dataloaders['train'])))
    return mols_smiles


def retrieve_qm9_smiles(dataset_info):
    dataset_name = dataset_info['name']
    if dataset_info['with_h']:
        pickle_name = dataset_name
    else:
        pickle_name = dataset_name + '_noH'

    file_name = 'qm9/temp/%s_smiles.pickle' % pickle_name
    try:
        with open(file_name, 'rb') as f:
            qm9_smiles = pickle.load(f)
        return qm9_smiles
    except OSError:
        try:
            os.makedirs('qm9/temp')
        except:
            pass
        qm9_smiles = compute_qm9_smiles(dataset_name, remove_h=not dataset_info['with_h'])
        with open(file_name, 'wb') as f:
            pickle.dump(qm9_smiles, f)
        return qm9_smiles


#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # Retrieve dataset smiles only for qm9 currently.
        if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
            self.dataset_smiles_list = retrieve_qm9_smiles(
                self.dataset_info)

    def compute_validity_and_connectivity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        connected = []
        valid_mols = []
        for graph in generated:
            mol = build_molecule(*graph, self.dataset_info)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                if len(mol_frags) == 1:
                    connected.append(True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)
        return valid, len(valid) / len(generated), len(connected) / len(generated), valid_mols

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)
    
    def compute_molecules_stability_from_graph(self, adjacency_matrices, numbers, charges, allowed_bonds=None,
        aromatic=True):
        if adjacency_matrices.ndim == 2:
            adjacency_matrices = adjacency_matrices.unsqueeze(0)
            numbers = numbers.unsqueeze(0)
            charges = charges.unsqueeze(0)

        if allowed_bonds is None:
            allowed_bonds = None

        if not aromatic:
            assert (adjacency_matrices == 1.5).sum() == 0 and (adjacency_matrices == 4).sum() == 0

        batch_size = adjacency_matrices.shape[0]
        stable_mask = torch.zeros(batch_size)
        n_stable_atoms = torch.zeros(batch_size)
        n_atoms = torch.zeros(batch_size)

        for i in range(batch_size):
            adj = adjacency_matrices[i]
            atom_nums = numbers[i]
            atom_charges = charges[i]

            mol_stable = True
            n_atoms_i, n_stable_i = 0, 0

            for j, (a_num, charge) in enumerate(zip(atom_nums, atom_charges)):
                if a_num.item() == 0:
                    continue
                row = adj[j]
                aromatic_count = int((row == 1.5).sum().item())
                normal_valence = float((row * (row != 1.5)).sum().item())
                combo = (aromatic_count, int(normal_valence))
                symbol = Chem.GetPeriodicTable().GetElementSymbol(int(a_num))
                allowed = allowed_bonds.get(symbol, {})

                # if _is_valid_valence_tuple(combo, allowed, int(charge)):
                #     n_stable_i += 1
                # else:
                #     mol_stable = False

                n_atoms_i += 1

            stable_mask[i] = float(mol_stable)
            n_stable_atoms[i] = n_stable_i
            n_atoms[i] = n_atoms_i

        return stable_mask, n_stable_atoms, n_atoms
    def compute_molecules_stability(self, valid_mols, aromatic=False, allowed_bonds=None):
        stable_list, stable_atoms_list, atom_counts_list = [], [], []

        
        for mol in valid_mols:
            if mol is None:
                continue
            n_atoms = mol.GetNumAtoms()
            adj = torch.zeros((1, n_atoms, n_atoms))
            numbers = torch.zeros((1, n_atoms), dtype=torch.long)
            charges = torch.zeros((1, n_atoms), dtype=torch.long)

            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                numbers[0, idx] = atom.GetAtomicNum()
                charges[0, idx] = atom.GetFormalCharge()

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()
                adj[0, i, j] = adj[0, j, i] = bond_type

            # stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            #     adj, numbers, charges, allowed_bonds, aromatic
            # )
            # stable_list.append(stable.item())
            # stable_atoms_list.append(stable_atoms.item())
            # atom_counts_list.append(atom_count.item())
            
    # def save_outliers
        
    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, connectivity, _ = self.compute_validity_and_connectivity(generated)
        print(f"""Validity over {len(generated)} molecules: {validity * 100 :.2f}%\n
              Connectivity over {len(generated)} molecules: {connectivity * 100 :.2f}%""")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
            connectivity = 0.0
        return [validity, uniqueness, novelty, connectivity], unique
    


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    except Chem.rdchem.KekulizeException:
        return None
    except Chem.rdchem.AtomValenceException:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(positions, atom_types, dataset_info):
    atom_decoder = dataset_info["atom_decoder"]
    X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol


def build_xae_molecule(positions, atom_types, dataset_info):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = dataset_info['atom_decoder']
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'jump':
                order = geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

if __name__ == '__main__':
    smiles_mol = 'C1CCC1'
    print("Smiles mol %s" % smiles_mol)
    chem_mol = Chem.MolFromSmiles(smiles_mol)
    block_mol = Chem.MolToMolBlock(chem_mol)
    print("Block mol:")
    print(block_mol)


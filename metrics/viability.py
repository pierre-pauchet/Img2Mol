
import sys
sys.path.append('/projects/iktos/pierre/CondGeoLDM/')  # Absolute path to project root
try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
    
import torch
import tqdm
import numpy as np
import scipy.stats as sp_stats

from qm9 import bond_analyze
from joblib import Parallel, delayed
from configs import datasets_config
import qm9.dataset as dataset 



BOND_DICT = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

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
        mol.AddBond(bond[0].item(), bond[1].item(), BOND_DICT[E[bond[0], bond[1]].item()])
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
                order = bond_analyze.get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'jump':
                order = bond_analyze.geom_predictor((atom_decoder[pair[0]], atom_decoder[pair[1]]), dists[i, j], limit_bonds_to_one=True)
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E

class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, dataset_smiles_list=None):
        self.atom_decoder = dataset_info['atom_decoder']
        self.dataset_smiles_list = dataset_smiles_list
        self.dataset_info = dataset_info

        # # Retrieve dataset smiles only for qm9 currently.
        # if dataset_smiles_list is None and 'qm9' in dataset_info['name']:
        #     self.dataset_smiles_list = retrieve_qm9_smiles(
        #         self.dataset_info)

    def compute_validity_and_connectivity(self, generated):
        """ generated: list of couples (positions, atom_types)
        Returns : valid smiles, valid rdkit mols, validity and connectivity values"""
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
                valid_mols.append(largest_mol)
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
    
    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, connectivity, valid_mols = self.compute_validity_and_connectivity(generated)
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
        return [validity, uniqueness, novelty, connectivity], valid_mols
    
def check_stability(positions, atom_type, dataset_info, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    atom_decoder = dataset_info['atom_decoder']
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            if dataset_info['name'] == 'qm9' or dataset_info['name'] == 'qm9_second_half' or dataset_info['name'] == 'qm9_first_half':
                order = bond_analyze.get_bond_order(atom1, atom2, dist)
            elif dataset_info['name'] == 'geom' or dataset_info['name'] == 'jump':
                order = bond_analyze.geom_predictor(
                    (atom_decoder[pair[0]], atom_decoder[pair[1]]), dist)
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_atoms = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = bond_analyze.allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) is int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_atoms += int(is_stable)

    molecule_stable = nr_stable_atoms == len(x)
    return molecule_stable, nr_stable_atoms, len(x)
    

def process_loader(dataloader):

    molecules = {'one_hot': [], 'x': [], 'node_mask': []}

    for batch in tqdm.tqdm(dataloader):
        o_h = batch["one_hot"].detach().cpu().float()
        pos = batch["positions"].detach().cpu()
        mask = batch["atom_mask"].detach().cpu().unsqueeze(2)
    
        B=batch["positions"].size(0)

        for i in range(B):
            molecules['one_hot'].append(o_h[i])
            molecules['x'].append(pos[i])
            molecules['node_mask'].append(mask[i])
            
    # molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}

    return molecules

def _check_single_molecule(mol, dataset_info):
    pos, atom_type = mol
    return check_stability(pos, atom_type, dataset_info)



def main_check_stability(remove_h: bool, batch_size=256):
    class Config:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 8
            self.remove_h = remove_h
            self.filter_n_atoms = None
            self.data_file = '/projects/iktos/pierre/CondGeoLDM/data/jump/charac_1_h.npy'
            self.dataset = 'jump'
            self.include_charges = True
            self.filter_molecule_size = None
            self.sequential = False
            self.percent_train_ds = 1
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.conditioning_type = "attention"
            

    cfg = Config()

    dataset_info = datasets_config.get_dataset_info(cfg.dataset, cfg.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)
    if use_rdkit:
        from qm9.rdkit_functions import BasicMolecularMetrics
        metrics = BasicMolecularMetrics(dataset_info)

    atom_decoder = dataset_info['atom_decoder']

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    
    
    train_data = process_loader(train_loader)
    test_data = process_loader(test_loader)
    train_dic, train_metrics = analyze_stability_for_molecules(train_data, dataset_info, parallel=True)
    test_dic, test_metrics = analyze_stability_for_molecules(test_data, dataset_info)
    if use_rdkit:
        print('For train')
        metrics.evaluate(train_data)
        print('For test')
        metrics.evaluate(test_data)
    # # else:
    # print('For train')
    # test_validity_for(train_data)
    # print('For test')
    # test_validity_for(test_data)
    print("train",train_dic)
    print("test",test_dic)


def analyze_stability_for_molecules(molecule_list, dataset_info, parallel=False):
    one_hot = molecule_list['one_hot']
    x = molecule_list['x']
    node_mask = molecule_list['node_mask']

    if isinstance(node_mask, torch.Tensor):
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [torch.sum(m) for m in node_mask]

    n_samples = len(x)

    molecule_stable = 0
    nr_stable_atoms = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))

    if parallel: 
        results = Parallel(n_jobs=64, backend="loky")(
            delayed(check_stability)(pos, atom_type, dataset_info)
            for (pos, atom_type) in tqdm.tqdm(processed_list, desc="Computing stability")
        )
            
        
        for is_stable, n_stable, total in results:
            molecule_stable += int(is_stable)
            nr_stable_atoms += int(n_stable)
            n_atoms += int(total)
    else : 
        for mol in tqdm.tqdm(processed_list, desc="Computing stability"):
            pos, atom_type = mol
            validity_results = check_stability(pos, atom_type, dataset_info)

            molecule_stable += int(validity_results[0])
            nr_stable_atoms += int(validity_results[1])
            n_atoms += int(validity_results[2])

    # Stability
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_atoms / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    if use_rdkit:
        metrics = BasicMolecularMetrics(dataset_info)
        rdkit_metrics, mols = metrics.evaluate(processed_list)
        #print("Unique molecules:", rdkit_metrics[1])
        return validity_dict, rdkit_metrics, mols
    else:
        return validity_dict, None, mols


# def analyze_node_distribution(mol_list, save_path):
#     hist_nodes = Histogram_discrete('Histogram # nodes (stable molecules)')
#     hist_atom_type = Histogram_discrete('Histogram of atom types')

#     for molecule in mol_list:
#         positions, atom_type = molecule
#         hist_nodes.add([positions.shape[0]])
#         hist_atom_type.add(atom_type)
#     print("Histogram of #nodes")
#     print(hist_nodes.bins)
#     print("Histogram of # atom types")
#     print(hist_atom_type.bins)
#     hist_nodes.normalize()
if __name__ == '__main__':

    main_check_stability(remove_h=False)
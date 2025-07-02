
import sys
sys.path.append('/projects/iktos/pierre/CondGeoLDM/')  # Absolute path to project root
try:
    from rdkit import Chem
    from qm9.rdkit_functions import BasicMolecularMetrics
    use_rdkit = True
except ModuleNotFoundError:
    use_rdkit = False
import qm9.dataset as dataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from qm9 import bond_analyze


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
    """ Mask atoms, return positions and atom types"""
    out = []
    for data in dataloader:
        for i in range(data['positions'].size(0)):
            positions = data['positions'][i].view(-1, 3)
            one_hot = data['one_hot'][i].view(-1, 5).type(torch.float32)
            mask = data['atom_mask'][i].flatten()
            positions, one_hot = positions[mask], one_hot[mask]
            atom_type = torch.argmax(one_hot, dim=1)
            out.append((positions, atom_type))
    return out


def main_check_stability(remove_h: bool, batch_size=32):
    from configs import datasets_config
    import qm9.dataset as dataset
    class Config:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = 8
            self.remove_h = True
            self.filter_n_atoms = None
            self.datadir = './data/jump/charac_30_h.npy'
            self.dataset = 'jump'
            self.include_charges = True
            self.filter_molecule_size = None
            self.sequential = False
            self.percent_train_ds = 5
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config()

    dataset_info = datasets_config.jump
    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)
    if use_rdkit:
        from qm9.rdkit_functions import BasicMolecularMetrics
        metrics = BasicMolecularMetrics(dataset_info)

    atom_decoder = dataset_info['atom_decoder']

    def test_validity_for(dataloader):
        count_mol_stable = 0
        count_atm_stable = 0
        count_mol_total = 0
        count_atm_total = 0
        for [positions, atom_types] in dataloader:
            is_stable, nr_stable, total = check_stability(
                positions, atom_types, dataset_info)

            count_atm_stable += nr_stable
            count_atm_total += total

            count_mol_stable += int(is_stable)
            count_mol_total += 1

            print(f"Stable molecules "
                  f"{100. * count_mol_stable/count_mol_total:.2f} \t"
                  f"Stable atoms: "
                  f"{100. * count_atm_stable/count_atm_total:.2f} \t"
                  f"Counted molecules {count_mol_total}/{len(dataloader)*batch_size}")

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    if use_rdkit:
        print('For test')
        metrics.evaluate(test_loader)
        print('For train')
        metrics.evaluate(train_loader)
    else:
        print('For train')
        test_validity_for(train_loader)
        print('For test')
        test_validity_for(test_loader)


def analyze_stability_for_molecules(molecule_list, dataset_info):
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

    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable += int(validity_results[0])
        nr_stable_atoms += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_atoms / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    if use_rdkit:
        metrics = BasicMolecularMetrics(dataset_info)
        rdkit_metrics = metrics.evaluate(processed_list)
        #print("Unique molecules:", rdkit_metrics[1])
        return validity_dict, rdkit_metrics
    else:
        return validity_dict, None


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

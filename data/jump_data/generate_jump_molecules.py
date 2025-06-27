import pandas as pd
import numpy as np
import tqdm 
from collections import defaultdict
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

parser = argparse.ArgumentParser(description="Generate molecular characteristics for JUMP dataset")
parser.add_argument('--num_conformers', type=int, default=10,)
parser.add_argument('--generate_with_h', action='store_true', help="Generate conformers with hydrogens")
args = parser.parse_args()

# Read the Parquet file
metadata = pd.read_parquet('/projects/iktos/pierre/CondGeoLDM/data/jump_data/metadata.parquet')
metadata.set_index(metadata.Metadata_InChI_ID, inplace=True, drop=True)
metadata = metadata.drop(columns=['Metadata_InChI_ID', 'Metadata_Is_dmso'])


# Load the embeddings from the .npy file
with open('/projects/iktos/pierre/CondGeoLDM/data/jump_data/Embeddings_norm.npy', 'rb') as f:
    embeddings = np.load(f)
embeddings = pd.DataFrame(embeddings, dtype=np.float16)

# Fuse embeddings and metadata
embeddings_list = [embeddings.iloc[i].tolist() for i in range(embeddings.shape[0])]
metadata['Embeddings'] = embeddings_list 

# Disable RDKit warnings (we generate without hydrogens, and it makes it very grumpy)
RDLogger.DisableLog('rdApp.warning')


def get_positions(mol_with_h, conf_id,generate_with_h=False):
    conf = mol_with_h.GetConformer(conf_id)
    pos = conf.GetPositions()  # returns (N_atoms, 3) ndarray directly
    if generate_with_h:
        return pos
    else: 
        symbols = [atom.GetSymbol() for atom in mol_with_h.GetAtoms()]
        mask = np.array([s != 'H' for s in symbols])
        return pos[mask]


def get_partial_charges(mol, valence_dict, i_mol):
    AllChem.ComputeGasteigerCharges(mol)

    charges = []
    result = AllChem.ComputeGasteigerCharges(mol)

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
        except KeyError:
            # RDKit did not assign a Gasteiger charge
            charge = float('nan')
        
        if np.isnan(charge):
            print(f"Gasteiger charge failed for atom {symbol} in molecule {i_mol}")
            return []  # Signal to skip molecule
        
        charges.append(charge)
        valence = atom.GetExplicitValence()
        valence_dict[symbol][round(charge)].add(valence)

    return charges


def clean_valence_dict(valence_dict):
    return {
        atom: {charge: sorted(list(valences)) for charge, valences in charge_dict.items()}
        for atom, charge_dict in valence_dict.items()
    }

    
# Generate molecular characteristics
def get_molecular_characteristics(metadata, num_conformers=1, generate_with_h=False):
    """
    Generate molecular characteristics for 10 conformers of molecules in the JUMP dataset,
    including charges, 3D positions, atom types, and embeddings.
    Generates a valence dictionnary containing allowed bonds for each atom type and formal charge.
    Returns a list of dictionaries with these characteristics and a list of IDs that failed 3D generation.
    """
    bugged_ids_during_3d_generation = []
    molecular_ids = metadata.Metadata_InChI.tolist()
    dataset_internal_ids = metadata.Metadata_JCP2022.apply(lambda x: x.split('_')[1]).tolist()
    conformers_characteristics = []
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    
    for i_mol, mol_inchi in tqdm.tqdm(enumerate(molecular_ids)):
        mol_with_h = Chem.MolFromInchi(mol_inchi, removeHs=False, sanitize=True)  #Supposedly, its better to generate 3D coordinates with hydrogens
        valence_dict = defaultdict(lambda: defaultdict(set))
        
        if mol_with_h is None:
            bugged_ids_during_3d_generation.append(mol_inchi)
            print(f"Invalid molecular identifier: {dataset_internal_ids[i_mol]}")
            continue
        
        # Generate 3D coordinates
        mol_with_h = Chem.AddHs(mol_with_h) #
        confs_ids = AllChem.EmbedMultipleConfs(mol_with_h, numConfs=num_conformers, params=params)
        if len(confs_ids) == 0:
            bugged_ids_during_3d_generation.append(mol_inchi)
            print(f"Failed to generate 3D coordinates for {dataset_internal_ids[i_mol]}")
            continue
        
        # Get positions without hydrogens
        if generate_with_h:
            positions = [get_positions(mol_with_h, conf_id, args.generate_with_h) for conf_id in confs_ids]
        else:
            # If we don't want hydrogens, we can remove them after embedding
            positions = [get_positions(mol_with_h, conf_id, args.generate_with_h) for conf_id in confs_ids]
            mol = Chem.RemoveHs(mol_with_h)  # Remove hydrogens after embedding
        
        # Get partial charges
        charges = get_partial_charges(mol_with_h, valence_dict, i_mol) if generate_with_h else get_partial_charges(mol, valence_dict, i_mol)
        if charges == []:
            continue  # Skip if charges could not be computed
        
        # Get atom types
        atom_types = np.array([atom.GetSymbol() for atom in mol_with_h.GetAtoms()]) if generate_with_h else np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
        
        # Cutoff molecules bigger than  atoms
        cutoff = 120 if generate_with_h else 70
        if len(charges) > cutoff:
            bugged_ids_during_3d_generation.append(mol_inchi)
            print(f"Too many atoms in {i_mol}, skipping.")
            continue
        
        charges = np.array(charges, dtype=np.float16)
        for _ in range(len(positions)):
            conformers_characteristics.append({
                "id": dataset_internal_ids[i_mol],
                "Inchi": mol_inchi,
                "charges": charges,
                "positions": positions[_],
                "atom_types": atom_types,
                "embeddings": metadata.iloc[i_mol].Embeddings,
            })
        
    # Fill valence dictionnary
    valence_dict_clean = clean_valence_dict(valence_dict)
    
    conformers_characteristics = sorted(conformers_characteristics, key=lambda x: len(x['charges']))
    return conformers_characteristics,  bugged_ids_during_3d_generation, valence_dict_clean

def main():
    charac, bugged_ids_during_3d_generation, allowed_bonds = get_molecular_characteristics(metadata, 
                                                                                           num_conformers=args.num_conformers, 
                                                                                           generate_with_h=args.generate_with_h)
    # Save charac to npy file
    np.save(f"./charac_{args.num_conformers}{'_h' if args.generate_with_h else ''}.npy", charac, allow_pickle=True)

    # Save bugged_ids_during_3d_generation to npy file
    np.save(f"./bugged_ids_during_3d_generation_{args.num_conformers}{'_h' if args.generate_with_h else ''}.npy", np.array(bugged_ids_during_3d_generation), allow_pickle=True)
    np.save(f"./allowed_bonds_{args.num_conformers}{'_h' if args.generate_with_h else ''}.txt", allowed_bonds, allow_pickle=True)
    
    
if __name__ == "__main__":
    print("Starting...'")
    main()

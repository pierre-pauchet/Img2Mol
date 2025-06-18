import pandas as pd
import os   
from pathlib import Path
import numpy as np
import tqdm 

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger


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
bugged_ids_during_3d_generation = []

# Generate molecular characteristics
def get_molecular_characteristics(metadata):
    molecular_ids = metadata.Metadata_InChI.tolist()
    dataset_internal_ids = metadata.Metadata_JCP2022.apply(lambda x: x.split('_')[1]).tolist()
    characteristics = []
    charges=[]
    for i, mol_id in tqdm.tqdm(enumerate(molecular_ids)):
        charges=[]
        mol = Chem.MolFromInchi(mol_id, removeHs=False, sanitize=True)
        if mol is None:
            bugged_ids_during_3d_generation.append(mol_id)
            print(f"Invalid molecular identifier: {mol_id}")
            continue
        
        # Get partial charges
        mol_with_h = Chem.AddHs(mol)
        result = AllChem.ComputeGasteigerCharges(mol_with_h)
        for atom in mol_with_h.GetAtoms():
            if atom.GetSymbol() != 'H':
                charge = float(atom.GetProp('_GasteigerCharge'))
                charges.append(charge)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            bugged_ids_during_3d_generation.append(mol_id)
            print(f"Failed to generate 3D coordinates for {mol_id}")
            continue
            
        # Get atomic positions
        conf = mol.GetConformer()
        positions = np.array([[pos.x, pos.y, pos.z] for pos in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))])

        # Get atom types
        atom_types = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])

        characteristics.append({
            "id": dataset_internal_ids[i],
            "Inchi": mol_id,
            "charges": np.array(charges),
            "positions": positions,
            "atom_types": atom_types,
            "embeddings": metadata.Embeddings[i],
        })
        del mol, mol_with_h, conf # Not sure if this is necessary
    return characteristics, bugged_ids_during_3d_generation

charac, bugged_ids_during_3d_generation = get_molecular_characteristics(metadata)


# Save charac to npy file
np.save("./charac.npy", charac, allow_pickle=True)

# Save bugged_ids_during_3d_generation to npy file
np.save("./bugged_ids_during_3d_generation.npy", np.array(bugged_ids_during_3d_generation), allow_pickle=True)

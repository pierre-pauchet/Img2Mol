import os 

from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import tqdm 
import numpy as np
import pandas as pd
# import tqdm
# charac = np.load('/projects/iktos/pierre/CondGeoLDM/charac.npy', allow_pickle=True)


# fingerprints_j = np.load("/projects/iktos/pierre/CondGeoLDM/data/jump_data/jump_fingerprints.npy", allow_pickle=True)
fingerprints_g = np.load("/projects/iktos/pierre/CondGeoLDM/data/jump_data/geom_fingerprints.npy", allow_pickle=True)
fingerprints_g = fingerprints_g
# fingerprints_j = fingerprints_j
similarities = []
best_hits = []
print('go')

# Paramètres de batches
batch_size = 1000  # ou 500, ou 2000 — à régler en fonction de vos contraintes mémoire

# List to collect the best hits
best_hits = []

# Initialise the best max similarities (excluding self-matches)
max_df = pd.DataFrame({'max_sim': np.zeros(len(fingerprints_g), dtype=np.float16)})

for start in tqdm.tqdm(range(0, len(fingerprints_g), batch_size)):
    end = min(start + batch_size, len(fingerprints_g))
    batch = fingerprints_g[start:end]

    # Calculate similarity between all and the batch
    sim = bulk_tanimoto_binary_similarity(batch, fingerprints_g)  # shape (N, B)

    # Remove self-similarity by masking diagonal block
    for i in range(end - start):  
        global_idx = start + i
        sim[i, global_idx] = -1.0
    # Take the max similarity across the batch (excluding self-comparison)
    batch_max = np.max(sim, axis=1)
    max_df.loc[start:end-1, 'max_sim'] = batch_max  

max_values = max_df['max_sim'].values
best_hits.append(sim)

# Sauvegarder le résultat
np.save('/projects/iktos/pierre/CondGeoLDM/data/jump_data/best_hits_geom.npy', max_values)
print('done')
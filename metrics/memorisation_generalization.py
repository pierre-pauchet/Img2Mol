import rdkit
from rdkit import Chem
from skfp.fingerprints import AtomPairFingerprint

import numpy as np
import os
from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
import matplotlib.pyplot as plt
import argparse
import tqdm
from qm9.rdkit_functions import smiles_to_mols, build_molecule, build_xae_molecule


def mols_to_fingerprints(mols_list, fp_size=1024, radius=2):
    atom_pair_fingerprint = AtomPairFingerprint(fp_size)
    
    fingerprints = atom_pair_fingerprint.transform(mols_list)
    return fingerprints
    
    
def compute_retrieval(fingerprints, train_fingerprints, threshold=0.8):
    """
    Compute retrieval metrics based on Tanimoto distances.
    Ie : How close the nearest neighbors of the newly generated molecules is to
    the training dataset
    """
    similarities = bulk_tanimoto_binary_similarity(fingerprints, train_fingerprints)
    max_similarities = np.max(similarities, axis=1)
    valid_similarities = max_similarities[max_similarities >= threshold]
    retrieval = np.sum(valid_similarities) / len(valid_similarities)

    return max_similarities, retrieval


def compute_self_similarity(fingerprints, batch_size=1000):
    """
    Compute self-similarity based on Tanimoto distances.
    Ie : How close the nearest neighbors of a dataset is to
    themselves
    """
    n = len(fingerprints)
    max_similarities = np.zeros(n, dtype=np.float16)

    for start in tqdm.tqdm(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch = fingerprints[start:end]

        sim = bulk_tanimoto_binary_similarity(batch, fingerprints)  # shape (batch_size, n)

        # Mask self-similarity
        for i in range(end - start):
            global_idx = start + i
            sim[i, global_idx] = -1.0  # Exclude self-similarity

        # Take max similarity for each item in the batch
        batch_max = np.max(sim, axis=1)

        # Store the values in the result array
        max_similarities[start:end] = batch_max

    return max_similarities


def compute_stacked_histogram(similarities_a, similarities_b, save_path, bins=100, label_a="Set A", label_b="Set B"):
    """
    Plot overlapping histograms of Tanimoto similarities for two sets.
    """
    plt.figure(figsize=(8, 5))

    # Plot both histograms with labels and transparency
    plt.hist(similarities_a, bins=bins, density=True, alpha=0.6, color='tab:blue', label=label_a, edgecolor='black', linewidth=0.5)
    plt.hist(similarities_b, bins=bins, density=True, alpha=0.6, color='tab:orange', label=label_b, edgecolor='black', linewidth=0.5)

    # Axis labels and title
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Density')
    plt.title('Distribution of Tanimoto Similarities')

    # Grid, legend, and layout
    plt.grid(axis='y', alpha=0.4)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    # Save and display
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sample mol with coditioning drawn from clusters')
    parser.add_argument('--train_fingerprints_file', type=str, default=None,
                        help='Path to the embeddings data directory')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Number of tries to find stable molecules')
    parser.add_argument('--data_file', type=str, default='/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy',
                        help='Conditioning type: geom, jump, or both')
    parser.add_argument('--model_path', type=str, default='CondGeoLDM/outputs')
    parser.add_argument('--save_path', type=str, default='CondGeoLDM/data/jump/figures')
    
    args = parser.parse_args()
    
    compute_retrieval(args, args.train_fingerprints_file, args.data_file, args.threshold)
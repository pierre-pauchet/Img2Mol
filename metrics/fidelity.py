import os
import sys
sys.path.append('/projects/iktos/pierre/CondGeoLDM/')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from rdkit import Chem
from skfp.fingerprints import ECFPFingerprint

import numpy as np
from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import argparse
import tqdm
from joblib import Parallel, delayed
import umap
import math


def save_fingerprints_file(fingerprints, save_path):
    np.save(save_path, fingerprints, allow_pickle=True)


def read_fingerprints_file(path, fp_size=1024):
    if not path.endswith(".npy"):
        path = path + ".npy"
    
    try:
        fingerprints = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{path}' not found ")
    except Exception as e:
        raise RuntimeError(f"Error while loading'{path}' : {e}")
    
    return fingerprints


def load_sdf_file(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    clean_mols = postprocess_rdkit_mols(mols)
    print(f"Loaded {len(clean_mols)} clean molecules from {sdf_path}")
    return mols
    

def postprocess_rdkit_mols(mols_list):
    """
    Postprocess RDKit molecules to ensure they are valid and sanitized.
    """
    valid_mols = []
    for mol in mols_list:
        try:
            Chem.SanitizeMol(mol)
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols = True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())  # Get the largest fragment
            valid_mols.append(largest_mol)
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
    return valid_mols


def rdkit_mols_to_fingerprints(mols_list, fp_size=1024, radius=2):
    atom_pair_fingerprint = ECFPFingerprint(fp_size)
    fingerprints = atom_pair_fingerprint.transform(mols_list)
    return fingerprints
    
    
def compute_top_k_retrieval(fingerprints, train_fingerprints, k=1, threshold=0.95):
    """
    Compute retrieval metrics based on Tanimoto distances.
    Ie : How close the nearest neighbors of the newly generated molecules is to
    the training dataset
    """
    print("Computing retrieval")
    similarities = bulk_tanimoto_binary_similarity(fingerprints, train_fingerprints)
    nearest_neighbours = np.sort(similarities, axis=1)[:, -k:][:, ::-1]
    nearest_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    mask = nearest_neighbours[:, 0] >= threshold # max values
    valid_similarities = nearest_neighbours[mask]
    retrieval = len(valid_similarities) / (len(nearest_neighbours)+1e-8)
    
    return nearest_neighbours, retrieval, nearest_indices


def compute_batch_top_k_retrieval(start, end, fingerprints, train_fingerprints, k=1, threshold=0.95):
    batch = fingerprints[start:end]
    similarities = bulk_tanimoto_binary_similarity(batch, train_fingerprints)
    nearest_neighbours = np.sort(similarities, axis=1)[:, -k:][:, ::-1]
    nearest_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    mask = nearest_neighbours[:, 0] >= threshold # max values
    valid_similarities = nearest_neighbours[mask]
    retrieval = len(valid_similarities) / (len(batch))
    
    return nearest_neighbours, retrieval, nearest_indices


def compute_top_k_retrieval_joblib(fingerprints, train_fingerprints, k=1, threshold=0.95, batch_size=100, n_jobs=-1):
    n = len(fingerprints)
    batch_size = min(batch_size, n)
    ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]

    results = Parallel(n_jobs=n_jobs-10)(
        delayed(compute_batch_top_k_retrieval)(start, end, fingerprints, train_fingerprints, k, threshold)
        for start, end in tqdm.tqdm(ranges)
    )

    all_nearest = []
    all_indices = []
    valid_total = 0
    total = 0

    for nearest, retrieval_batch, indices in results:
        all_nearest.append(nearest)
        all_indices.append(indices)
        valid_total += int(retrieval_batch * len(nearest))
        total += len(nearest)

    retrieval = valid_total / total
    nearest_neighbours = np.concatenate(all_nearest, axis=0)
    nearest_indices = np.concatenate(all_indices, axis=0)

    return nearest_neighbours, retrieval, nearest_indices
    
def compute_self_similarity(fingerprints, batch_size=1000):
    """
    Compute self-similarity based on Tanimoto distances.
    Ie : How close the nearest neighbors of a dataset is to
    themselves
    """
    print("Computing self-similarity")
    n = len(fingerprints)
    max_similarities = np.zeros(n, dtype=np.float16)
    batch_size = min(batch_size, n)
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

def compute_batch_self_similarity(start, end, fingerprints):
    
    batch = fingerprints[start:end]
    sim = bulk_tanimoto_binary_similarity(batch, fingerprints)
    
    for i in range(end - start):
        sim[i, start + i] = -1.0  # Mask self-similarity
    return np.max(sim, axis=1)


def compute_self_similarity_joblib(fingerprints, batch_size=1000, n_jobs=-1):
    n = len(fingerprints)
    batch_size = min(batch_size, n)
    ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_batch_self_similarity)(start, end, fingerprints) for start, end in tqdm.tqdm(ranges)
    )

    return np.concatenate(results)


def compute_stacked_histogram(similarities_list, labels, save_path, bins=100, title=""):
    plt.figure(figsize=(8, 5))

    cmap = plt.get_cmap('tab10')
    n = len(similarities_list)
    x_min = min(np.min(s) for s in similarities_list)
    x_max = max(np.max(s) for s in similarities_list)
    x = np.linspace(x_min, x_max, 1000)
    assert len(labels) == n, "Number of labels must match number of similarity lists"
    text_lines = []
    for i, similarities in enumerate(similarities_list):
        color = cmap(i)
        counts, _, _ = plt.hist(
            similarities,
            bins=bins,
            density=True,
            stacked=False,
            alpha=0.15,
            color=color,
            label=labels[i],
            edgecolor='black',
            linewidth=0.5
        )
        try:
            kde = gaussian_kde(similarities)
        except np.linalg.LinAlgError:
            print(f"Skipped KDE for label {labels[i]} due to singular matrix")
            continue
        y_vals = kde(x)
        plt.plot(x, y_vals, color=color, lw=2)

        mean = np.average(x, weights=y_vals)
        std = np.sqrt(np.average((x - mean) ** 2, weights=y_vals))

        y_mean = kde(mean)
        text = f'μ = {mean:.3f}\nσ = {std:.3f}'
        plt.text(mean, y_mean + np.max(y_vals)/5, text, color=color, fontsize=9, ha='center', va='bottom')

    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Density')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    
    
def compute_stacked_umap(fingerprints_list, labels, save_path, title='', n_neighbors=15, min_dist=0.1):
    assert len(fingerprints_list) == len(labels), "Number of fingerprints lists must match number of labels"

    # Concatenate and track indices
    all_fps = np.concatenate(fingerprints_list, axis=0)
    group_sizes = [len(fps) for fps in fingerprints_list]
    group_indices = np.cumsum([0] + group_sizes)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(all_fps)

    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    n = len(fingerprints_list)
    fig, axes = plt.subplots(1 + (n-1)//3, n, figsize=(6 * n, 5), squeeze=False)
    cmap = plt.get_cmap('tab20') if n > 10 else plt.get_cmap('tab10')

     # Grid size
    n = len(fingerprints_list)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    cmap = plt.get_cmap('tab20') if n > 10 else plt.get_cmap('tab10')

    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            start, end = group_indices[i], group_indices[i + 1]
            ax.scatter(
                embedding[start:end, 0],
                embedding[start:end, 1],
                s=10,
                color=cmap(i),
                alpha=0.7
            )
            ax.set_title(labels[i])
            ax.set_xlabel('UMAP-1')
            ax.set_ylabel('UMAP-2')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.3)
        else:
            ax.axis('off')  # Empty subplot
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # la place pour le suptitle
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description='Sample mol with coditioning drawn from clusters')
    parser.add_argument('--train_fp_file', type=str, default="/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/jump_fingerprints.npy",
                        help='Path to the embeddings data directory')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Number of tries to find stable molecules')
    parser.add_argument('--fp_file', type=str, default='/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy',
                        help='Conditioning type: geom, jump, or both')
    parser.add_argument('--model_path', type=str, default='CondGeoLDM/outputs')
    parser.add_argument('--save_path_figures', type=str, default='CondGeoLDM/data/jump/figures')
    args = parser.parse_args()
    
    
    sdf_paths={
        # 'semla' : "/projects/iktos/pierre/sampled_mols/semla-flow/predictions.smol.sdf",
        # "flowmol" : "/projects/iktos/pierre/sampled_mols/flowmol/50k_new_molecules_geom.sdf",
        # "gcdm" : "/projects/iktos/pierre/sampled_mols/GCDM/all_batches_geom.sdf",
        # "eqgat" : "/projects/iktos/pierre/sampled_mols/EQGAT‑diff/geom.sdf",
        # # "geoldm" : "/projects/iktos/pierre/sampled_mols/GeoLDM/geom.sdf",
        # # "jumpxatt" : "/projects/iktos/pierre/sampled_mols/jump/1000xatt.sdf",
        # # "jumpvanilla" : "/projects/iktos/pierre/sampled_mols/jump/1000vanilla.sdf",
        # "jumpphen0" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed0.sdf",
        # "jumpphen1" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed1.sdf",
        # "jumpphen2" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed2.sdf",
        "jumpphen19" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed19.sdf",
    }
    
    fingerprints = {
        name: rdkit_mols_to_fingerprints(load_sdf_file(path), fp_size=1024)
        for name, path in sdf_paths.items() if path is not None
    }

    train_fp = read_fingerprints_file(args.train_fp_file)
    mask = np.random.choice(len(train_fp), size=50000, replace=False)
    train_fp_50k = train_fp[mask]
    # fingerprints = {
    #     "semla_nopp": read_fingerprints_file("/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/fp_semla_from_geom_no_pp.npy"),
    # #     "flowmol": read_fingerprints_file("/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/flowmol_fp_from_geom.npy"),
    # #     "gcdm": read_fingerprints_file("/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/gcdm_fp_from_geom.npy"),
    # #     "eqgat": read_fingerprints_file("/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/eqgat_fp_from_geom.npy"),
    # }
    mallats = {
        name: compute_top_k_retrieval_joblib(fp, train_fp, k=3, batch_size=75, threshold=args.threshold)
        for name, fp in fingerprints.items()
    }
    
    output_dir = Path("/import/pr_iktos/pierre/CondGeoLDM/data/fingerprints_data")

    for name, fp in fingerprints.items():
        # Sauvegarde des empreintes
        fp_path = output_dir / f"fp_{name}_from_jump.npy"
        save_fingerprints_file(fp, str(fp_path))

        # Sauvegarde des best hits (résultat du top-k)
        hits, _, _ = mallats[name]
        hits_path = output_dir / f"best_hits_{name}.npy"
        np.save(hits_path, hits)
    # ss, _, a = compute_top_k_retrieval_joblib(train_fp_50k, train_fp, k=4, batch_size=100, n_jobs=-1)
    # np.save( '/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/best_hits_geom_50k.npy', ss[:,1:])

    
    
if __name__ == '__main__':
    main()
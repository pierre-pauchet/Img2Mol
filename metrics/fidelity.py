import os
import sys
sys.path.append('/projects/iktos/pierre/CondGeoLDM/')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from skfp.fingerprints import ECFPFingerprint

from collections import Counter
import numpy as np
from skfp.distances.tanimoto import bulk_tanimoto_binary_similarity
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
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
    suppl = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]
    # clean_mols = postprocess_rdkit_mols(mols)
    # print(f"Loaded {len(clean_mols)} clean molecules from {sdf_path}")
    print(f"Loaded {len(mols)} DIRTY molecules from {sdf_path}")
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
            Chem.AddHs(largest_mol)  
            valid_mols.append(largest_mol)
        except Exception as e:
            print(f"Error sanitizing molecule: {e}")
            
    return valid_mols


def rdkit_mols_to_fingerprints(mols_list, fp_size=1024, radius=2):
    atom_pair_fingerprint = ECFPFingerprint(fp_size, radius)
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
    retrieval = len(valid_similarities) / (len(nearest_neighbours))
    
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
    
    
def compute_stacked_umap(fingerprints_list, labels, save_path, title='', use_umap=True, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    assert len(fingerprints_list) == len(labels), "Number of fingerprints lists must match number of labels"
    
    # Concatenate and track indices
    all_fps = np.concatenate(fingerprints_list, axis=0)
    group_sizes = [len(fps) for fps in fingerprints_list]
    group_indices = np.cumsum([0] + group_sizes)
    
    # Create group labels for each point
    group_labels = []
    for i, size in enumerate(group_sizes):
        group_labels.extend([i] * size)
    group_labels = np.array(group_labels)
    
    
    # Apply UMAP without shuffling to maintain group correspondence
    if use_umap:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(all_fps)
    else:
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(all_fps)        
    
    # Get embedding bounds for consistent scaling
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    # Grid configuration
    n = len(fingerprints_list)
    cols = 3
    rows = math.ceil(n / cols)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    cmap = plt.get_cmap('tab20') if n > 10 else plt.get_cmap('tab10')
    
    if not use_umap:
        print("Explained variance ratios:", pca.explained_variance_ratio_)
        print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

    # Plot each group
    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            start, end = group_indices[i], group_indices[i + 1]
            ax.scatter(
                embedding[start:end, 0],
                embedding[start:end, 1],
                s=10,
                color=cmap(i),
                alpha=0.7,
                label=labels[i]
            )
            center_x = np.mean(embedding[start:end, 0])
            center_y = np.mean(embedding[start:end, 1])
            ax.scatter(
                center_x, 
                center_y, 
                marker='x', 
                s=100, 
                color=cmap(i),
                edgecolors='black',
                linewidths=3,
                label='Center of Mass'
            )
            
            ax.set_title(labels[i], fontsize=12)
            ax.set_xlabel('UMAP-1')
            ax.set_ylabel('UMAP-2')
            ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
            ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
            ax.grid(alpha=0.3)
        else:
            ax.axis('off') 
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return embedding, group_labels


def compute_stacked_umap_with_overlay(fingerprints_list, labels, save_path, title='', n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Enhanced version that includes an overlay plot showing all groups together
    """
    assert len(fingerprints_list) == len(labels), "Number of fingerprints lists must match number of labels"
    
    # Concatenate and track indices
    all_fps = np.concatenate(fingerprints_list, axis=0)
    group_sizes = [len(fps) for fps in fingerprints_list]
    group_indices = np.cumsum([0] + group_sizes)
    
    # Create group labels for each point
    group_labels = []
    for i, size in enumerate(group_sizes):
        group_labels.extend([i] * size)
    group_labels = np.array(group_labels)
    
    print(f"Data type: {type(all_fps)}")
    print(f"Shape: {all_fps.shape}")
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(all_fps)
    
    # Get embedding bounds
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    # Grid configuration (add 1 for overlay plot)
    n = len(fingerprints_list)
    total_plots = n + 1  # +1 for overlay
    cols = 3
    rows = math.ceil(total_plots / cols)
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    
    cmap = plt.get_cmap('tab20') if n > 10 else plt.get_cmap('tab10')
    
    # Plot overlay first
    ax = axes[0]
    for i in range(n):
        start, end = group_indices[i], group_indices[i + 1]
        ax.scatter(
            embedding[start:end, 0],
            embedding[start:end, 1],
            s=8,
            color=cmap(i),
            alpha=0.6,
            label=labels[i]
        )
    ax.set_title('All Groups Overlay', fontsize=12, fontweight='bold')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot individual groups
    for i in range(n):
        ax = axes[i + 1]
        start, end = group_indices[i], group_indices[i + 1]
        ax.scatter(
            embedding[start:end, 0],
            embedding[start:end, 1],
            s=10,
            color=cmap(i),
            alpha=0.7
        )
        ax.set_title(labels[i], fontsize=12)
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return embedding, group_labels

def count_molecular_properties(sdf_paths, labels, verbose=False):
    assert len(labels) == len(sdf_paths), "Labels must match the number of SDF files."
    
    results = {}
    
    for i, sdf_path in enumerate(sdf_paths):
        label = labels[i]
        print(f"Processing {label}...")
        RDLogger.DisableLog('rdApp.error')
        RDLogger.DisableLog('rdApp.warning')
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True) 
        mols = [mol for mol in suppl if mol is not None]

        # Compute mean number of atoms
        num_atoms = [mol.GetNumAtoms() for mol in mols]
        mean_num_atoms = np.mean(num_atoms)
        std_num_atoms = np.std(num_atoms)
        
        # Compute mean logP
        logps = [Descriptors.MolLogP(mol) for mol in mols]
        mean_logp = np.mean(logps)
        std_logp = np.std(logps)

        # Atom type repartition
        atom_types = []
        for mol in mols:
            atom_types.extend([atom.GetSymbol() for atom in mol.GetAtoms()])
        atom_type_counts = Counter(atom_types)
        
        # Convert atom counts to percentages
        total_atoms = len(atom_types)
        atom_type_percentages = {atom: (count/total_atoms)*100 for atom, count in atom_type_counts.items()}
        
        # Store results in dictionary
        results[label] = {
            'num_molecules': len(mols),
            'atoms': {
                'mean': mean_num_atoms,
                'std': std_num_atoms
            },
            'logP': {
                'mean': mean_logp,
                'std': std_logp
            },
            'atom_types': {
                'counts': dict(atom_type_counts),
                'percentages': atom_type_percentages
            }
        }
        
        if verbose :
            print(f"Mean number of atoms: {mean_num_atoms:.2f} ± {std_num_atoms:.2f}")
            print(f"Mean logP: {mean_logp:.2f}±{std_logp:.2f}")
            print("Atom type repartition:")
            for atom, count in atom_type_counts.items():
                percentage = atom_type_percentages[atom]
                print(f"  {atom}: {percentage:.2f}% ({count})")

    print("="*80)
    print(f"{'Dataset':<20} {'Molecules':<10} {'Atoms (μ±σ)':<15} {'LogP (μ±σ)':<15} {'Top Atom':<10}")
    print("-"*80)
    for label in labels:
        data = results[label]
        atoms_summary = f"{data['atoms']['mean']:.1f} ± {data['atoms']['std']:.1f}"
        logp_summary = f"{data['logP']['mean']:.2f} ± {data['logP']['std']:.2f}"
        
        # Find most common atom type
        top_atom_type = max(data['atom_types']['percentages'].items(), 
                           key=lambda x: x[1])
        top_atom = f"{top_atom_type[0]} ({top_atom_type[1]:.1f}%)"
        
        print(f"{label:<20} {data['num_molecules']:<10} {atoms_summary:<15} "
              f"{logp_summary:<15} {top_atom:<10}")
    
   # Extract data for plotting
    num_molecules = [results[label]['num_molecules'] for label in labels]
    atoms_mean = [results[label]['atoms']['mean'] for label in labels]
    atoms_std = [results[label]['atoms']['std'] for label in labels]
    logp_mean = [results[label]['logP']['mean'] for label in labels]
    logp_std = [results[label]['logP']['std'] for label in labels]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    y_positions = np.arange(len(labels))
    
    # Plot 1: Number of molecules (horizontal bar chart)
    bars = ax1.barh(y_positions, num_molecules, color=colors, alpha=0.7)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Number of Molecules')
    ax1.set_title('Number of Molecules per Dataset')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, num_molecules)):
        ax1.text(value + max(num_molecules) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value}', va='center', fontsize=10)
    
    # Plot 2: Number of atoms with error bars and horizontal dashes
    ax2.errorbar(atoms_mean, y_positions, xerr=atoms_std, fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    ax2.scatter(atoms_mean, y_positions, c=colors, s=100, marker='+', linewidths=3)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Mean Number of Atoms')
    ax2.set_title('Mean Number of Atoms per Molecule (with std)')
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: LogP with error bars and horizontal dashes
    ax3.errorbar(logp_mean, y_positions, xerr=logp_std, fmt='none', 
                ecolor='black', capsize=5, capthick=2)
    ax3.scatter(logp_mean, y_positions, c=colors, s=100, marker='+', linewidths=3)
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('Mean LogP')
    ax3.set_title('Mean LogP (with std)')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Top atom type percentages
    top_atom_percentages = []
    top_atom_names = []
    for label in labels:
        top_atom_type = max(results[label]['atom_types']['percentages'].items(), 
                           key=lambda x: x[1])
        top_atom_percentages.append(top_atom_type[1])
        top_atom_names.append(top_atom_type[0])
    
    bars = ax4.barh(y_positions, top_atom_percentages, color=colors, alpha=0.7)
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels([f"{label}\n({atom})" for label, atom in zip(labels, top_atom_names)])
    ax4.set_xlabel('Percentage (%)')
    ax4.set_title('Most Common Atom Type Percentage')
    ax4.grid(axis='x', alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar, value, atom) in enumerate(zip(bars, top_atom_percentages, top_atom_names)):
        ax4.text(value + max(top_atom_percentages) * 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # # Save plot if path provided
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     print(f"Plot saved to: {save_path}")
    
    plt.show()
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='Sample mol with conditioning drawn from clusters')
    parser.add_argument('--dataset', type=str, default="geom")
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Number of tries to find stable molecules')
    parser.add_argument('--fp_file', type=str, default='/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy',
                        help='Conditioning type: geom, jump, or both')
    parser.add_argument('--model_path', type=str, default='CondGeoLDM/outputs')
    parser.add_argument('--save_path_figures', type=str, default='CondGeoLDM/data/jump/figures')
    args = parser.parse_args()
    
    train_fp_file = f"/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/{args.dataset}_fingerprints.npy"
    
    sdf_paths={
        'semla' : "/projects/iktos/pierre/sampled_mols/semla-flow/predictions.smol.sdf",
        "flowmol" : "/projects/iktos/pierre/sampled_mols/flowmol/50k_new_molecules_geom.sdf",
        "gcdm" : "/projects/iktos/pierre/sampled_mols/GCDM/all_batches_geom.sdf",
        "eqgat" : "/projects/iktos/pierre/sampled_mols/EQGAT‑diff/geom.sdf",
        "geoldm" : "/projects/iktos/pierre/sampled_mols/GeoLDM/geom.sdf",
        "geoldm_ogmodelprep" : "/projects/iktos/pierre/sampled_mols/GeoLDM/ogmodel_prep.sdf",
        "debug" : "/projects/iktos/pierre/sampled_mols/GeoLDM/debug.sdf",
        # "jumpxatt" : "/projects/iktos/pierre/sampled_mols/jump/1000xatt.sdf",
        # "jumpvanilla" : "/projects/iktos/pierre/sampled_mols/jump/1000vanilla.sdf",
        # "jumpphen0" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed0.sdf",
        # "jumpphen1" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed1.sdf",
        # "jumpphen2" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed2.sdf",
        # "jumpphen19" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed19.sdf",
        # "jumpphen12" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed12.sdf",
        # "jumpphen14" : "/projects/iktos/pierre/sampled_mols/jump/250xattfixed14.sdf",
        # "jump_unsanitized" : "/projects/iktos/pierre/CondGeoLDM/data/jump/raw_data/jump_molecules.sdf"
    }
    
    fingerprints = {
        name: rdkit_mols_to_fingerprints(load_sdf_file(path), fp_size=1024, radius=3)
        for name, path in sdf_paths.items() if path is not None
    }

    train_fp = read_fingerprints_file(train_fp_file)
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
        fp_path = output_dir / f"fp_{name}_from_{args.dataset}.npy"
        save_fingerprints_file(fp, str(fp_path))

        # Sauvegarde des best hits (résultat du top-k)
        hits, _, _ = mallats[name]
        hits_path = output_dir / f"best_hits_{name}.npy"
        np.save(hits_path, hits)
    ss, _, a = compute_top_k_retrieval_joblib(train_fp_50k, train_fp, k=4, batch_size=100, n_jobs=-1)
    np.save( f'/projects/iktos/pierre/CondGeoLDM/data/fingerprints_data/best_hits_{args.dataset}_50k.npy', ss[:,1:])


    
    
if __name__ == '__main__':
    main()
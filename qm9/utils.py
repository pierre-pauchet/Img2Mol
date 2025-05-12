import torch
import pandas as pd
import numpy as np
import os

def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name == "qm9":
        return compute_mean_mad_from_dataloader(dataloaders["train"], properties)
    elif dataset_name == "qm9_second_half" or dataset_name == "qm9_second_half":
        return compute_mean_mad_from_dataloader(dataloaders["valid"], properties)
    else:
        raise Exception("Wrong dataset name")


def compute_mean_mad_from_dataloader(dataloader, properties):
    property_norms = {}
    for property_key in properties:
        print("Computing mean and mad for property:", property_key)
        values = dataloader.dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        property_norms[property_key] = {}
        property_norms[property_key]["mean"] = mean
        property_norms[property_key]["mad"] = mad
    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(
        charges.shape[:2] + (-1,)
    )
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms, verbose=False):
    batch_size, n_nodes, _ = minibatch["positions"].size()
    node_mask = minibatch["atom_mask"].unsqueeze(2)
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]["mean"]) / property_norms[key]["mad"]  # normalization
        if len(properties.size()) == 1:
            # Global feature.
            assert properties.size() == (batch_size,)
            reshaped = properties.view(batch_size, 1, 1).repeat(1, n_nodes, 1)  # for every property
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            assert properties.size()[:2] == (batch_size, n_nodes)
            context_key = properties
            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)
            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError("Invalid tensor size, more than 3 axes.")
    # Concatenate
    context = torch.cat(context_list, dim=2)  # doesn't change vectors if properties.size == 1
    # Mask disabled nodes!
    context = context * node_mask
    assert context.size(2) == context_node_nf
    return context


def load_embeddings(embeddings_data_dir, merge_data=False):
    """
    Creates the array for embeddings
    returns :
    - embeddings : torch.Tensor (n_embeddings, n_features_embeddings)
    - metadata : df.Dataframe (n_embeddings, 3)
        columns : ['Metadata_JCP2022', 'Metadata_InChI', 'Metadata_Is_dmso']
    """
    print("Loading metadata from", embeddings_data_dir)
    metadata = pd.read_parquet(embeddings_data_dir+'/'+"metadata.parquet")
    metadata.set_index(metadata.Metadata_InChI_ID, inplace=True, drop=True)
    print("Loading embeddings from", embeddings_data_dir)
    
    embeddings = np.load(embeddings_data_dir+'/'+"Embeddings_norm.npy")
    if (merge_data):  # make a unique DF with embeddings and corresponding molecule metadata
        embeddings = pd.DataFrame(embeddings)
        for i in range(embeddings.shape[1]):
            if i == 0 or i == 100 or i == 1535:
                print(i)
            metadata[f"Embeddings_{i}"] = embeddings.iloc[:, 1]
    print("Done loading embeddings")
    return torch.from_numpy(embeddings), metadata


def prepare_embeddings(embeddings, metadata, minibatch, verbose=False):
    """ """
    batch_size, n_nodes, _ = minibatch["positions"].size()
    node_mask = minibatch["atom_mask"].unsqueeze(2)
    batch_embeddings = embeddings[:batch_size]
    context = batch_embeddings.unsqueeze(1)
    context = context.expand(batch_size, n_nodes, embeddings.shape[1])
    if verbose:
        print(f"Emeddings cont shape: {context.shape}")
        print(f"Embeddings as context: {context}")
    context = context * node_mask
    context = context.unsqueeze(2)
    
    print("Context shape:", context.shape)
    print('DONE')
    return context
    if verbose:
        print(f"batch_size: {batch_size}, n_nodes: {n_nodes}")
        print(f"node_mask shape: {node_mask.shape}")

    return None

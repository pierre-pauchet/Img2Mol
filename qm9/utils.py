import torch
import pandas as pd
import numpy as np
import os
from torch.profiler import schedule, profile, record_function, ProfilerActivity
from train_test import train_epoch
def compute_mean_mad(dataloaders, properties, dataset_name):
    if dataset_name in ["qm9", "jump"]:
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





def prepare_embeddings(minibatch, verbose=False):
    """ """
    batch_size, n_nodes, _ = minibatch["positions"].size()
    
    node_mask = minibatch["atom_mask"].unsqueeze(2)
    context = minibatch["embeddings"]
    
    if verbose:
        print(f"Emeddings cont shape: {context.shape}")
        print(f"Embeddings as context: {context}")
    context = context * node_mask
    # context = context.unsqueeze(2)
    
    print("Context shape:", context.shape)
    print('DONE')
    return context


def trace_handler(p):
    for device in ["cuda", 'cpu']:
        sort_by_keyword = "self_" + device + "_time_total"
        output = p.key_averages().table(sort_by=sort_by_keyword, 
                                        row_limit=10,)
        print(output)    
        print(f"Trace handler called at step {p.step_num}")
        # Save to a plain .txt file
        log_path = os.path.join("./log/", f"profile_step_{p.step_num}_{device}.txt")
        with open(log_path, "w") as f:
            f.write(output)

        detailed_log_path = os.path.join("./log/", f"profile_step_{p.step_num}_mul_stacktrace.txt")
        print(f"[Profiler] Saved readable profile to: {log_path}")
        with open(detailed_log_path, "w") as f:
            for evt in p.events():
                if "aten::mul" in evt.name:
                    dev = "cuda" if evt.device_type == torch.profiler.ProfilerActivity.CUDA else "cpu"
                    f.write(f"[aten::mul] device: {dev}, cpu_time: {evt.cpu_time_total}ns, cuda_time: {evt.cuda_time_total}ns\n")
                    f.write("Stack trace:\n")
                    for frame in evt.stack:
                        f.write(f"  {frame}\n")
                    f.write("\n")
        
def profiler(args, loader, epoch, model, model_dp,
                    model_ema, ema, device, dtype, property_norms,
                    nodes_dist, dataset_info,
                    gradnorm_queue, optim, prop_dist, test_loaders):
    """ 
    """
    my_schedule = schedule(
        skip_first=10,
        wait=5,
        warmup=1,
        active=3,
        repeat=2)
    with profile(schedule=my_schedule,
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=trace_handler
    ) as prof:
        with record_function("model_inference"):
            print("Profiling your code")
            train_epoch(args=args, loader=loader, epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, test_loaders=test_loaders,
                    prof=prof)       
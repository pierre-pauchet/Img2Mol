from rdkit import Chem
import numpy as np
from os.path import join
import pandas as pd
import skfp

import argparse
import pickle
import util
from equivariant_diffusion.utils import assert_correctly_masked
from qm9.sampling import sample_chain, sample
from qm9.analyze import check_stability
import qm9.visualizer as vis
from qm9.models import get_latent_diffusion
from configs.datasets_config import get_dataset_info
from qm9 import dataset, losses
import torch


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def save_and_sample_chain(args, eval_args, device, flow, n_tries, n_nodes, dataset_info, 
                          sample_id=0, num_visualisations=10,):
    
    for i in range(num_visualisations):
        target_path = f"eval/chain{i}/"

        one_hot, charges, x, node_mask = sample_chain(
            args, device, flow, n_tries, dataset_info, test_loaders=None
        )
        for sample_id in range(one_hot.size(0)):
            vis.save_xyz_file(
                join(eval_args.model_path, f"{target_path}"),
                one_hot[sample_id],
                charges[sample_id],
                x[sample_id],
                dataset_info,
                sample_id,
                name="chain",
                node_mask=node_mask[sample_id],
            )

        # Check that the mask is correct
        vis.visualize_chain(join(eval_args.model_path, f"{target_path}"), dataset_info, spheres_3d=True)

    # Legacy code :

    # for i in range(num_chains):
    #     target_path = f'eval/chain_{i}/'

    #     one_hot, charges, x = sample_chain(
    #         args, device, flow, n_tries, dataset_info)

    #     vis.save_xyz_file(
    #         join(eval_args.model_path, target_path), one_hot, charges, x,
    #         dataset_info, sample_id, name='chain')

    #     vis.visualize_chain_uncertainty(
    #         join(eval_args.model_path, target_path), dataset_info,
    #         spheres_3d=True)
    return one_hot, charges, x


def sample_different_sizes(
    args,
    eval_args,
    device,
    generative_model,
    nodes_dist,
    dataset_info,
    n_samples=10,
    save=False,
):
    nodesxsample = nodes_dist.sample(n_samples)
    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info, nodesxsample=nodesxsample
    )

    if save:
        vis.save_xyz_file(
            join(eval_args.model_path, "eval/molecules/"),
            one_hot,
            charges,
            x,
            sample_id=0,
            name="molecule",
            dataset_info=dataset_info,
            node_mask=node_mask,
        )
    return one_hot, charges, x, node_mask


def sample_only_stable_different_sizes(
    args,
    eval_args,
    device,
    flow,
    nodes_dist,
    dataset_info,
    n_samples=10,
    n_tries=50,
    save=False,
):
    assert n_tries > n_samples

    nodesxsample = nodes_dist.sample(n_tries)
    one_hot, charges, x, node_mask = sample(
        args, device, flow, dataset_info, nodesxsample=nodesxsample
    )

    counter = 0
    stable_mol_list = []
    for i in range(n_tries):
        num_atoms = int(node_mask[i : i + 1].sum().item())
        atom_type = (
            one_hot[i : i + 1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        )
        x_squeeze = x[i : i + 1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = n_tries - i - 1
        num_remaining_samples = n_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print("Found stable mol.")
                if save:
                    vis.save_xyz_file(
                        join(eval_args.model_path, "eval/molecules/"),
                        one_hot[i : i + 1],
                        charges[i : i + 1],
                        x[i : i + 1],
                        sample_id=counter,
                        name="molecule_stable",
                        dataset_info=dataset_info,
                        node_mask=node_mask[i : i + 1],
                    )
            stable_mol_list.append(
                (
                    one_hot[i : i + 1],
                    charges[i : i + 1],
                    x[i : i + 1],
                    node_mask[i : i + 1],
                )
            )
            counter += 1

            if counter >= n_samples:
                break
    if len(stable_mol_list) > 0:
        one_hot_stable = torch.cat([item[0] for item in stable_mol_list], dim=0)
        charges_stable = torch.cat([item[1] for item in stable_mol_list], dim=0)
        x_stable = torch.cat([item[2] for item in stable_mol_list], dim=0)
        node_mask_stable = torch.cat([item[3] for item in stable_mol_list], dim=0)
    else:
        one_hot_stable = torch.empty(
            (0, one_hot.shape[1], one_hot.shape[2]), device=one_hot.device
        )
        charges_stable = torch.empty((0, charges.shape[1]), device=charges.device)
        x_stable = torch.empty((0, x.shape[1], x.shape[2]), device=x.device)
        node_mask_stable = torch.empty((0, node_mask.shape[1]), device=node_mask.device)

    return one_hot_stable, charges_stable, x_stable, node_mask_stable


def graph_to_mol(one_hot, charges, x, node_mask):
    molecules = {"one_hot": [], "x": [], "node_mask": []}
    for i in range(one_hot.shape[0]):
        molecules["one_hot"].append(one_hot.detach().cpu())
        molecules["x"].append(x.detach().cpu())
        molecules["node_mask"].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}


def main():
    parser = argparse.ArgumentParser(
        description="Sample mol with conditioning drawn from clusters"
    )

    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--n_tries",
        type=int,
        default=50,
        help="Number of tries to find stable molecules",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy",
        help="Conditioning type: geom, jump, or both",
    )
    parser.add_argument("--model_path", type=str, default="CondGeoLDM/outputs")
    parser.add_argument("--stable_only", type=bool, default="False")
    parser.add_argument("--visualise_chain", type=bool, default="False")
    # Parse the arguments
    eval_args, _ = parser.parse_known_args()
    print(eval_args)
    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve dataset info and dataloaders
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    flow, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, dataloaders["train"]
    )
    # if prop_dist is not None:
    #     property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    #     prop_dist.set_normalizer(property_norms)
    flow.to(device)

    fn = "generative_model_ema.npy" if args.ema_decay > 0 else "generative_model.npy"
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    flow.load_state_dict(flow_state_dict)

    # 1. Sample molecules from trained models.
    if eval_args.stable_only:
        one_hot, charges, x, node_mask = sample_only_stable_different_sizes(
            args,
            eval_args,
            device,
            flow,
            nodes_dist,
            dataset_info,
            n_samples=eval_args.n_samples,
            n_tries=eval_args.n_tries,
            save=False,
        )
    else:
        one_hot, charges, x, node_mask = sample_different_sizes(
            args,
            eval_args,
            device,
            flow,
            nodes_dist,
            dataset_info,
            n_samples=eval_args.n_samples,
            save=eval_args.save,
        )

    if not eval_args.save:
        print("Need args.save==True to visualize sampled molecules")
    else:
        print("Visualizing molecules.")
        vis.visualize(
            join(eval_args.model_path, "eval/molecules/"),
            dataset_info,
            max_num=100,
            spheres_3d=True,
        )

    # 2. Visualise some new chains (optional)
    if eval_args.visualise_chain:
        print("Sampling visualization chain.")
        save_and_sample_chain(
            args,
            eval_args,
            device,
            flow,
            n_tries=eval_args.n_tries,
            n_nodes=eval_args.n_nodes,
            dataset_info=dataset_info,
        )

    # 3. Compute Viability metrics
    stability_dict, rdkit_metrics = analyze_and_save(
        args,
        eval_args,
        device,
        generative_model,
        nodes_dist,
        prop_dist,
        dataset_info,
        n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen,
        save_to_xyz=eval_args.save_to_xyz,
    )


if __name__ == "__main__":
    main()

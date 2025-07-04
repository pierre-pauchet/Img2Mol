import numpy as np
import torch
import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.size(0) == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = torch.tensor(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).float()
    Qx = torch.tensor(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).float()
    Qy = torch.tensor(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).float()

    Q = torch.mm(torch.mm(Qz, Qx), Qy)

    Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        # print(z_x.size(), Q.size())
        new_x = torch.matmul(z_x.view(-1, 3), Q.T).view(1, -1, 3)
        # print(new_x.size())
        new_z = torch.cat([new_x, z_h], dim=2)
        results.append(new_z)

    results = torch.cat(results, dim=0)
    return results


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None, 
                 test_loaders=None, random_idx=False, n_samples=4):
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom' :
        n_nodes = 30
    elif args.dataset == 'jump' and args.remove_h:
        n_nodes = 30
    elif args.dataset == 'jump' and not args.remove_h:
        n_nodes = 44
    else:
        raise ValueError()

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes).unsqueeze(1).unsqueeze(0)
        context = context.repeat(1, n_nodes, 1).to(device)
        #context = torch.zeros(n_samples, n_nodes, args.context_node_nf).to(device)
    else:
        context = None
    if args.conditioning_mode == 'cross_attention':
        # samples phenotypes for cross-attention conditioning from the test loaders
        all_phenotypes = []
        for batch in test_loaders:
            emb = batch['embeddings'].clone()
            all_phenotypes.append(emb)

        all_phenotypes = torch.cat(all_phenotypes, dim=0).to(device)
        if random_idx:
                random_embedding_idx = torch.randint(0, all_phenotypes.size(0), (n_samples,))
                phenotypes = all_phenotypes[random_embedding_idx]
        else:
            phenotypes = all_phenotypes[:n_samples]
    else: 
        phenotypes = None
        
    # node_mask = torch.ones(n_samples, n_nodes, 1).to(device)
    # edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
    # edge_mask = edge_mask.repeat(n_samples, 1, 1).view(-1, 1).to(device)
    nodesxsample = [n_nodes - (i*5) for i in range(n_samples)]
    nodesxsample = torch.tensor(nodesxsample)
    node_mask = torch.zeros(n_samples, max(nodesxsample))
    for i in range(n_samples):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(n_samples * max(nodesxsample) * max(nodesxsample), 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    keep_frames = min(40, args.n_stability_samples)
    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, phenotypes, keep_frames=keep_frames)

            # chain = chains[:chains.size(0) //n_samples] # take only the first chain
            chain = reverse_tensor(chain)
        
            chain = chain.permute(1,0,2,3) # (n_samples, n_frames, n_nodes, 4 + args.include_charges)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[:, -1:, :, :].repeat(1, 10, 1, 1)], dim=1)
            x = chain[:, -1:, :, 0:3]
            one_hot = chain[:, -1:, :, 3:-1]
            one_hot = torch.argmax(one_hot, dim=3)

            atom_type = one_hot[0].squeeze(0).cpu().detach().numpy()
            x_squeeze = x[0].squeeze(0).cpu().detach().numpy()
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            # Prepare entire chain.
            x = chain[:, :, :, 0:3]
            one_hot = chain[:, :, :, 3:-1]
            one_hot = F.one_hot(torch.argmax(one_hot, dim=3), num_classes=len(dataset_info['atom_decoder']))
            charges = torch.round(chain[:, :, :, -1:]).long()
            node_mask_final = node_mask.unsqueeze(1).repeat(1, chain.size(1), 1, 1).to(device)
            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError

    return one_hot, charges, x, node_mask_final


def sample(args, device, generative_model, dataset_info,
           prop_dist=None, test_loaders=None, nodesxsample=torch.tensor([10]), context=None, phenotypes=None,
           fix_noise=False, random_idx=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    # TODO FIX: This conditioning just zeros.
    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None
    if args.conditioning_mode == 'cross_attention':
        # samples phenotypes for cross-attention conditioning from the test loaders
        all_phenotypes = []
        for batch in test_loaders:
            emb = batch['embeddings'].clone()
            all_phenotypes.append(emb)

        all_phenotypes = torch.cat(all_phenotypes, dim=0).to(device)
        if random_idx:
                random_embedding_idx = torch.randint(0, all_phenotypes.size(0), (batch_size,))
                phenotypes = all_phenotypes[random_embedding_idx]
        else:
            phenotypes = all_phenotypes[:batch_size]
    else: 
        phenotypes = None
    
    if args.probabilistic_model == 'diffusion':
        x, h = generative_model.sample(batch_size, max_n_nodes, node_mask, edge_mask, context, phenotypes, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.float(), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.float(), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask

def sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = torch.tensor([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = torch.tensor(np.linspace(min_val, max_val, n_frames)).unsqueeze(1)
        context.append(context_row)
    context = torch.cat(context, dim=1).float().to(device)

    one_hot, charges, x, node_mask = sample(args, device, generative_model, dataset_info, prop_dist, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask

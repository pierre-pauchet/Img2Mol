# TODO : ajouter gestion des devices,pour cibler ls GPU du cluster
# idee : checker tous les assignements de device et les remplacer par un device unique, défini en dessous des argsparser

# Rdkit import should be first, do not move it
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Set CUDA_VISIBLE_DEVICES to use GPU 

try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import os
from configs.datasets_config import jump
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import torchinfo
import pickle
from qm9.utils import prepare_context, compute_mean_mad, prepare_embeddings, profiler
from train_test import train_epoch, test, analyze_and_save


parser = argparse.ArgumentParser(description="E3Diffusion")
parser.add_argument("--exp_name", type=str, default="")


# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', default=True,
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# VAE args
parser.add_argument('--latent_nf', type=int, default=2,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='jump',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='data/jump_data',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,help='save model')
parser.add_argument('--num_workers', type=int, default=4, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--resume', type=str, default=None,
                    help='')

parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
# Mine
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--conditioning_mode', type=str, default='original',
                    help='original | naive | attention | other') #maybe i should default to a no cond mode
parser.add_argument('--data_file', type=str, default='/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy')
parser.add_argument('--filter_molecule_size', type=int, default=None)
parser.add_argument('--sequential', type=bool, default=False)
parser.add_argument('--percent_train_ds', type=float, default=None,
                    help='number of molecules used in train.')
parser.add_argument('--viability_metrics_epochs', type=int, default=None,
                    help='Frequence of computation of metrics. Defaults to test_epochs')

args = parser.parse_args()


if args.resume is not None:
    resume = args.resume
    
    with open(join(args.resume, "args.pickle"), "rb") as f:
        args = pickle.load(f)
    exp_name = args.exp_name + "_resume"
    start_epoch = args.start_epoch
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method


    args.resume = resume
    args.break_train_epoch = False

    args.batch_size = 32
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    # Careful with this -->
    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = normalization_factor
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = aggregation_method


dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info["atom_encoder"]
atom_decoder = dataset_info["atom_decoder"]

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device

if args.viability_metrics_epochs is None:
    args.viability_metrics_epochs = args.test_epochs

dtype = torch.float32



utils.create_folders(args)

# Wandb config
if args.no_wandb:
    mode = "disabled"
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': vars(args),
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save("*.txt")


# Build Jump Dataset
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

atom_encoder = dataset_info["atom_encoder"]
atom_decoder = dataset_info["atom_decoder"]

data_dummy = next(iter(dataloaders["train"]))

# Conditioning
if len(args.conditioning) == 0:
    context_node_nf = 0
    property_norms = None
else:
    if args.conditioning_mode == "original":  # OG Physics conditoning
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
    elif args.conditioning_mode == "naive":  # Naive conditioning
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        embeddings = prepare_embeddings(data_dummy, verbose=True)
        context_node_nf = embeddings.size(2)
    elif args.conditioning_mode in ['cross_attention', 'other']: # Attention conditioning
        context_node_nf = 0
        property_norms = None
    elif args.conditioning_mode in ['other']: # Attention conditioning
            raise NotImplementedError('Attention conditioning not implemented yet')
    else: # No conditioning
        context_node_nf = 0
        property_norms = None


args.context_node_nf = context_node_nf


# Create Latent Diffusion Model or Audoencoder

if args.train_diffusion:
    model, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, dataloaders["train"]
    )
else:
    model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info, dataloaders['train'])

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    print(args)
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, "generative_model_ema.npy"))
        optim_state_dict = torch.load(join(args.resume, "flow.npy"))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e10
    # best_nll_test = 1e1
    test_loaders = dataloaders['test']
    
    ## Profiling 
    # profiler(args=args, loader=dataloaders["train"], epoch=0, model=model, model_dp=model_dp,
    #                 model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
    #                 nodes_dist=nodes_dist, dataset_info=dataset_info,
    #                 gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, test_loaders=test_loaders)
    # print("Done profiling")
    
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info, prof=None,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, 
                    test_loaders=test_loaders)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        if isinstance(model, en_diffusion.EnVariationalDiffusion):
            wandb.log(model.log_info(), commit=True)
        if epoch % args.test_epochs == 0:
            if not args.break_train_epoch and args.train_diffusion and epoch % args.viability_metrics_epochs == 0:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples,
                                 test_loaders=test_loaders)

            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)


            if nll_val < best_nll_val or epoch == 0:
                best_nll_val = nll_val
                # best_nll_test = nll_test

                # save best model over previous best
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/flow.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)
                        
            #save current model
            # if args.save_model:
            #     utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
            #     utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
            #     if args.ema_decay > 0:
            #         utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
            #     with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
            #         pickle.dump(args, f)
            print("Val loss: %.4f" % (nll_val))
            print("Best val loss: %.4f" % best_nll_val)
            wandb.log({"Val loss ": nll_val, "Epoch": epoch}, commit=True)
            # wandb.log({"Best cross-validated test loss ": best_nll_test, "Epoch":epoch}, commit=True)
            
        nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
        partition='Test', device=device, dtype=dtype,
        nodes_dist=nodes_dist, property_norms=property_norms)
        print('Test loss:  %.4f' % nll_test)
        wandb.log({"Test loss ": nll_test, "Epoch":epoch}, commit=True)
        


if __name__ == "__main__":
    print("Starting...'")
    main()

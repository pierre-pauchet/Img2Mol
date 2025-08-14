#!/bin/bash
#SBATCH --job-name=TravailGPU # name of job
#SBATCH --output=TravailGPU%j.out # output file (%j = job ID)
#SBATCH --error=TravailGPU%j.err # error file (%j = job ID)
#SBATCH --constraint=a100 # reserve GPUs with 80 GB
#SBATCH --nodes=2 # reserve 2 node
#SBATCH --ntasks=16 # reserve 16 tasks (or processes)
#SBATCH --gres=gpu:8 # reserve 8 GPUs
#SBATCH --cpus-per-task=10 # reserve 10 CPUs per task (and associated memory)
#SBATCH --time=20:00:00 # maximum allocation time "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # deactivate hyperthreading
#SBATCH --account=ufc43hj@a100 # V100 accounting

# module purge # purge modules inherited by default
# conda deactivate # deactivate environments inherited by default
# module load pytorch-gpu/py3/2.3.0 # load modules

# set -x # activate echo of launched commands
srun python main_jump.py --dataset jump --n_epochs 40 --n_stability_samples 100 --diffusion_steps 1000 \
--batch_size 78 --nf 256 --n_layers 6 --lr 3e-4 --num_workers 8 --test_epochs 1 --model egnn_dynamics \
--visualize_every_batch 10000 --datadir $SCRATCH \
--trainable_ae --train_diffusion --online False \
--latent_nf 2 --percent_train_ds 100 --exp_name jump_xatt_h_6layers_40e --viability_metrics_epochs 1 \
--data_file data/jump/charac_30_h.npy --conditioning_mode attention


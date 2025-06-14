# I AM CURRETLY WORKING ON MODIFYING THE FOLLOWING PROJECT : A COMPLETE README WILL BE ADDED IN A LATER COMMIT. MEANWHILE, CHECK OUT THE ORIGINAL CODE FROM THE AUTHORS BELOW .

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoLDM/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2305.01140-B31B1B.svg)](https://arxiv.org/abs/2305.01140)

<!-- [[Code](https://github.com/MinkaiXu/GeoLDM)] -->

![cover](equivariant_diffusion/framework.png)

Official code release for the paper "Geometric Latent Diffusion Models for 3D Molecule Generation", accepted at *International Conference on Machine Learning, 2023*.

## Environment

Install the required packages from `requirements.txt`. A simplified version of the requirements can be found [here](https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/requirements.txt).

**Note**: If you want to set up a rdkit environment, it may be easiest to install conda and run:
``conda create -c conda-forge -n my-rdkit-env rdkit`` and then install the other required packages. But the code should still run without rdkit installed though.


## Train the GeoLDM

### For QM9

```python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9```

### For Drugs

First follow the intructions at `data/geom/README.md` to set up the data.

```python main_geom_drugs.py --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 --train_diffusion --trainable_ae --latent_nf 2 --exp_name geoldm_drugs```

**Note**: In the paper, we present an encoder early-stopping strategy for training the Autoencoder. However, in later experiments, we found that we can even just keep the encoder untrained and only train the decoder, which is faster and leads to similar results. Our released version uses this strategy. This phenomenon is quite interesting and we are also still actively investigating it.

### Pretrained models

We also provide pretrained models for both QM9 and Drugs. You can download them from [here](https://drive.google.com/drive/folders/1EQ9koVx-GA98kaKBS8MZ_jJ8g4YhdKsL?usp=sharing). The pretrained models are trained with the same hyperparameters as the above commands except that latent dimensions `--latent_nf` are set as 2 (the results should be roughly the same if as 1). You can load them for running the following evaluations by putting them in the `outputs` folder and setting the argument `--model_path` to the path of the pretrained model `outputs/$exp_name`.

## Evaluate the GeoLDM

To analyze the sample quality of molecules:

```python eval_analyze.py --model_path outputs/$exp_name --n_samples 10_000```

To visualize some molecules:

```python eval_sample.py --model_path outputs/$exp_name --n_samples 10_000```

Small note: The GPUs used for these experiment were pretty large. If you run out of GPU memory, try running at a smaller size.
<!-- The main reason is that the EGNN runs with fully connected message passing, which becomes very memory intensive. -->

## Conditional Generation

### Train the Conditional GeoLDM

```python main_qm9.py --exp_name exp_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning alpha --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1```

The argument `--conditioning alpha` can be set to any of the following properties: `alpha`, `gap`, `homo`, `lumo`, `mu` `Cv`. The same applies to the following commands that also depend on alpha.

### Generate samples for different property values

```python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --property alpha --n_sweeps 10 --task qualitative```

### Evaluate the Conditional GeoLDM with property classifiers

#### Train a property classifier
```cd qm9/property_prediction```  
```python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha --model_name egnn```

Additionally, you can change the argument `--model_name egnn` by `--model_name numnodes` to train a classifier baseline that classifies only based on the number of nodes.

#### Evaluate the generated samples

Evaluate the trained property classifier on the samples generated by the trained conditional GeoLDM model

```python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --classifiers_path qm9/property_prediction/outputs/exp_class_alpha --property alpha  --iterations 100  --batch_size 100 --task edm```

## Citation
Please consider citing the our paper if you find it helpful. Thank you!
```
@inproceedings{xu2023geometric,
  title={Geometric Latent Diffusion Models for 3D Molecule Generation},
  author={Minkai Xu and Alexander Powers and Ron Dror and Stefano Ermon and Jure Leskovec},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgements

This repo is built upon the previous work [EDM](https://arxiv.org/abs/2203.17003). Thanks to the authors for their great work!
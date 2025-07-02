import rdkit
from rdkit import Chem
import numpy as np
import os
import pandas as pd
import skfp

import argparse


def 










def main():
    parser = argparse.ArgumentParser(description='Sample mol with coditioning drawn from clusters')

    parser.add_argument('--embeddings_data_file', type=str, default=None,
                        help='Path to the embeddings data directory')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--n_tries', type=int, default=50,
                        help='Number of tries to find stable molecules')
    parser.add_argument('--data_file', type=str, default='/projects/iktos/pierre/CondGeoLDM/data/jump/charac_30_h.npy',
                        help='Conditioning type: geom, jump, or both')
    parser.add_argument('--model_path', type=str, default='CondGeoLDM/outputs')


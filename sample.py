from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from pylab import figure, axes, scatter, title, show

#from rdkit import Chem
#from rdkit.Chem import Draw

LATENT_DIM = 292
TARGET = 'autoencoder'

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--target', type=str, default=TARGET,
                        help='What model to sample from: autoencoder, encoder, decoder.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--do_conv_encoder', type=bool, metavar='True', default=True,
                        help='Whether to use a convolutional or recurrent encoder.')
    return parser

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)
    
def read_charset(filename):
    h5f = h5py.File(filename, 'r')
    charset =  h5f['charset'][:]
    h5f.close()
    return charset
    
def save_latent_data(filename, charset, x_latent):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('charset', data = charset)
    h5f.create_dataset('latent_vectors', data = x_latent)
    h5f.close()

def autoencoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False, do_conv_encoder=args.do_conv_encoder)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim, do_conv_encoder=args.do_conv_encoder)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.autoencoder.predict(data[0].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)

def decoder(args, model=MoleculeVAE(), verbose=False):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    for line in data:
        sampled = model.decoder.predict(line.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        print(sampled)

def encoder(args, model=MoleculeVAE()):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim, do_conv_encoder=args.do_conv_encoder)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data)
    if args.save_h5:
        save_latent_data(args.save_h5, charset, x_latent)
    else:
        np.savetxt(sys.stdout, x_latent, delimiter = '\t')
    return x_latent

def main():
    parser = get_arguments()
    args = parser.parse_args()
    model = MoleculeVAE()

    if args.target == 'autoencoder':
        autoencoder(args, model)
    elif args.target == 'encoder':
        encoder(args, model)
    elif args.target == 'decoder':
        decoder(args, model)

if __name__ == '__main__':
    main()

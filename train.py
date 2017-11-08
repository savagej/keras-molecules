from __future__ import print_function

import argparse
import os
import h5py
import numpy as np


NUM_EPOCHS = 1
BATCH_SIZE = 600
LATENT_DIM = 292
RANDOM_SEED = 1337

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
    parser.add_argument('model', type=str,
                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
                        help='Number of samples to process per minibatch during training.')
    parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
                        help='Seed to use to start randomizer for shuffling.')
    conv_parser = parser.add_mutually_exclusive_group(required=False)
    conv_parser.add_argument('--do_conv_encoder', dest='do_conv_encoder', action='store_true')
    conv_parser.add_argument('--do_recu_encoder', dest='do_conv_encoder', action='store_false')
    parser.set_defaults(do_conv_encoder=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Turn on tensorboard storage of info. Needs small test data size.')

    return parser.parse_args()

def main():
    args = get_arguments()
    np.random.seed(args.random_seed)

    from molecules.model import MoleculeVAE
    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    
    data_train, data_test, charset = load_dataset(args.data)
    model = MoleculeVAE()
    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = args.latent_dim, 
                     do_conv_encoder=args.do_conv_encoder)
    else:
        model.create(charset, latent_rep_size = args.latent_dim, 
                     do_conv_encoder=args.do_conv_encoder)

    checkpointer = ModelCheckpoint(filepath = args.model,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
    
    tensorboarder = TensorBoard(histogram_freq=1, embeddings_freq=1)
    if args.tensorboard:
        callbacks_list = [checkpointer, reduce_lr, tensorboarder]
    else:
        callbacks_list = [checkpointer, reduce_lr]

    model.autoencoder.fit(
        data_train,
        data_train,
        shuffle = True,
        epochs = args.epochs,
        batch_size = args.batch_size,
        callbacks = callbacks_list,
        validation_data = (data_test, data_test)
    )

if __name__ == '__main__':
    main()

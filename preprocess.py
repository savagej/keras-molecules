import argparse
import pandas
import h5py
import numpy as np
from molecules.utils import one_hot_array, one_hot_index, one_hot_array_fast
import functools

from sklearn.model_selection import train_test_split

MAX_NUM_ROWS = 500000
SMILES_COL_NAME = 'structure'

def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('infile', type=str, help='Input file name')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--length', type=int, metavar='N', default = MAX_NUM_ROWS,
                        help='Maximum number of rows to include (randomly sampled).')
    parser.add_argument('--smiles_column', type=str, default = SMILES_COL_NAME,
                        help="Name of the column that contains the SMILES strings. Default: %s" % SMILES_COL_NAME)
    parser.add_argument('--property_column', type=str,
                        help="Name of the column that contains the property values to predict. Default: None")
    parser.add_argument('--full_file', type=str,
                        help="Full file without train_test split")
    parser.add_argument('--test_size', type=float, default=0.2,
                        help="Size of test set, see sklearn.model_selection.train_test_split for details")
 
    return parser

def chunk_iterator(dataset, chunk_size=1000):
    chunk_indices = np.array_split(np.arange(len(dataset)),
                                    len(dataset)/chunk_size)
    for chunk_ixs in chunk_indices:
        chunk = dataset[chunk_ixs]
        yield (chunk_ixs, chunk)
    raise StopIteration
    
def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                         chunk_size=1000, apply_fn=None):
    new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                     chunks=tuple([chunk_size]+list(dataset_shape[1:])))
    for (chunk_ixs, chunk) in chunk_iterator(dataset, chunk_size=chunk_size):
        if not apply_fn:
            new_data[chunk_ixs, ...] = chunk
        else:
            new_data[chunk_ixs, ...] = apply_fn(chunk)

def get_data(args):    
    data = pandas.read_csv(args.infile)
    keys = data[args.smiles_column].map(len) < 121

    if args.length <= len(keys):
        data = data[keys].sample(n = args.length)
    else:
        data = data[keys]

    structures = data[args.smiles_column].map(lambda x: list(x.ljust(120)))
    structures = structures.apply(lambda lst: [lett.encode('utf8') for lett in lst])

    if args.property_column:
        properties = data[args.property_column][keys]

    del data

    charset = list(functools.reduce(lambda x, y: set(y) | x, structures, set()))
    return structures, charset

def write_full_file(args, structures, charset, chunk_size=1000):
    h5f_full = h5py.File(args.full_file, 'w')
    h5f_full.create_dataset('charset', data = charset)
    def one_hot_fn(ch):
        return np.array([[one_hot_array_fast(i, charset) for i in x] for x in structures[ch]])
    create_chunk_dataset(h5f_full, 'data_test', structures.index,
                         (len(structures.index), 120, len(charset)),
                         chunk_size=chunk_size, apply_fn=one_hot_fn)
    structures = [ b''.join(x) for x in structures]

    h5f_full.create_dataset('structures', data = structures)
    if args.property_column:
        h5f_full.create_dataset('property_train', data = properties[structures.index])
    h5f_full.close()
    return None

def main():
    parser = get_arguments()
    args = parser.parse_args()
    structures, charset = get_data(args)
    test_size = args.test_size if args.test_size < 1 else int(args.test_size)
    
    
    train_idx, test_idx = map(np.array,
                              train_test_split(structures.index, test_size = test_size))

    h5f = h5py.File(args.outfile, 'w')
    h5f.create_dataset('charset', data = charset)
    
    chunk_size = min([len(train_idx), len(test_idx), 1000]) / 2

    def create_chunk_dataset(h5file, dataset_name, dataset, dataset_shape,
                             chunk_size=1000, apply_fn=None):
        new_data = h5file.create_dataset(dataset_name, dataset_shape,
                                         chunks=tuple([chunk_size]+list(dataset_shape[1:])))
        for (chunk_ixs, chunk) in chunk_iterator(dataset, chunk_size=chunk_size):
            if not apply_fn:
                new_data[chunk_ixs, ...] = chunk
            else:
                new_data[chunk_ixs, ...] = apply_fn(chunk)

    def one_hot_fn(ch):
        return np.array([[one_hot_array_fast(i, charset) for i in x] for x in structures[ch]])
    
    create_chunk_dataset(h5f, 'data_train', train_idx,
                         (len(train_idx), 120, len(charset)),
                         chunk_size=chunk_size, apply_fn=one_hot_fn)
    create_chunk_dataset(h5f, 'data_test', test_idx,
                         (len(test_idx), 120, len(charset)),
                         chunk_size=chunk_size, apply_fn=one_hot_fn)
    
    if args.property_column:
        h5f.create_dataset('property_train', data = properties[train_idx])
        h5f.create_dataset('property_test', data = properties[test_idx])
    h5f.close()
    
    if args.full_file:
        write_full_file(args, structures, charset)
                            

       

if __name__ == '__main__':
    main()

import gzip
import pandas
import h5py
import numpy as np

def one_hot_array(i, n):
    return [1 if ix == i else 0 for ix in range(n)]

def one_hot_index(vec, charset):
    return [charset.index(x) for x in vec]

def one_hot_array_fast(i, charset):
    return [1 if ix == i else 0 for ix in charset]

def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0, ):
        return None
    return int(oh[0][0])

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x].decode("utf8"), vec)).strip()

def load_dataset(filename, split = True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, charset)
    else:
        return (data_test, charset)

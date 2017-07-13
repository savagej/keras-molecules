# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:27:37 2017

@author: John Savage
"""
import time
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

import h5py


def get_arguments():
    parser = argparse.ArgumentParser(description='Read encoded h5')
    parser.add_argument('infile', type=str, help='Input file name')
    parser.add_argument('structurefile', type=str, help='Structures file name')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--compress', type=int, metavar='N', default = 2,
                        help='Dimensions to reduce to for plotting using SVD')
    parser.add_argument('--plot_only', action="store_true",
                        help="Only create plots, don't write to csv")
    parser.add_argument('--figure_folder', type=str, default = ".",
                        help='Folder to place the figures')

    return parser.parse_args()

def plot_features_2D(df_vis, structures, figure_folder="."):
    print("Plotting features")
    df_vis.plot.scatter(0, 1)
    plt.savefig(os.path.join(figure_folder, "features.png"))
    # Plot with color of peptide length
    
    df_vis["len"] = structures.str.len()
    
    df_vis.plot.scatter(0, 1, c="len")
    plt.savefig(os.path.join(figure_folder, "features_with_length.png"))

def main(start_time):
    args = get_arguments()

    # Read input file
    f = h5py.File(args.infile, 'r')
    
    df_latent = pd.DataFrame(f['latent_vectors'][:])
    f.close()
    
    print("file read: {}".format(time.time()-start_time))
    
    # Read file containing sequences
    f2 = h5py.File(args.structurefile,'r')
    print("file2 read: {}".format(time.time()-start_time))
    
    df_structures = pd.DataFrame(f2['structures'][:], columns=["structures"])
    f2.close()
    structures = df_structures["structures"].apply(lambda x: x.decode('utf8'))\
                    .astype(str).str.strip()
    print("structures decoded: {}".format(time.time()-start_time))
    
    # Reduce dimensions for visualisation
    
    if args.compress > 0:
        svd = TruncatedSVD(args.compress)
        vis = svd.fit_transform(df_latent)
        df_vis = pd.DataFrame(vis)
        print("file compressed: {}".format(time.time()-start_time))
        print("explained ratio: {}".format(svd.explained_variance_ratio_))
    else:
        df_vis = df_latent.copy()
    
    # Output csv of latent features for use elsewhere
    if args.plot_only:
        print("Not outputting csv: {}".format(time.time()-start_time))
    else:
        print("Outputting csv: {}".format(time.time()-start_time))
        df_latent["structures"] = structures  
        df_latent.to_csv(args.outfile, index=False)
       
    plot_features_2D(df_vis, structures, args.figure_folder)
    return 0

if __name__ == "__main__":
    start_time = time.time()
    main(start_time)
    print("done: {}".format(time.time()-start_time))
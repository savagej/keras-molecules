# Create csv of input file with sequence column named "structure"
tcell_compact_less.csv

# activate keras2 env in linux
source activate keras2
# preprocess data
python preprocess.py data/tcell_compact_less.csv data//tcell_compact_processed2.h5 --full_file data/tcell_compact_all.h5


# activate tensorflow env in windows
activate tensorflow
# train model
python train.py data/tcell_compact_processed2.h5 models/tcell_compact_model.h5 --epochs 20 --batch_size 100 --latent_dim 100 # 1 epoch ~= 2000s

# encode peptide strings
python sample.py data/tcell_compact_all.h5 ./models/tcell_compact_model.h5 --save_h5 data/tcell_compact_encoded.h5 --latent_dim 100 --target encoder

# decode cluster centers
python sample.py data/cluster_4.h5 ./models/tcell_compact_model.h5 --latent_dim 100 --target decoder
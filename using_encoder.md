# Create csv of input file with sequence column named "structure"
data/peptide_sequences.csv

# activate keras env (or whatever you've called your conda environment)
source activate keras
# preprocess data
python preprocess.py data/peptide_sequences.csv data/peptide_sequences_processed.h5 --full_file data/peptide_sequences_all.h5

# encode peptide strings
python sample.py data/peptide_sequences_all.h5 ./models/mutated_sequences2_model_epochX50.h5 --save_h5 data/peptide_sequences_encoded.h5 --latent_dim 100 --target encoder

# Read encoded features from h5 to csv and plot 2d figures
 python read_encoded_features.py --compress 2 data/peptide_sequences_encoded.h5 data/peptide_sequences_all.h5 data/peptide_sequences_encoded.csv
import scanpy as sc
import pandas as pd
import os
import numpy as np


input_folder = 'dataset_norm'
output_folder = 'dataset_H'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        adata = sc.read_csv(file_path)

        adata.var_names_make_unique()

        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable']]

        col_sums = np.sum(adata.X, axis=0)
        sorted_indices = np.argsort(col_sums)[::-1]
        adata = adata[:, sorted_indices]

        output_filename = filename.replace('LogNormalize', 'H')
        output_path = os.path.join(output_folder, output_filename)

        adata.to_df().to_csv(output_path)

print("All files done", output_folder)
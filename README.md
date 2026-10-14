# scGEN: Single-Cell Gene-Aware Embedded Network

[![Paper](https://img.shields.io/badge/Advanced%20Science-10.1002%2Fadvs.77746-blue)](https://doi.org/10.1002/advs.77746)
[![Data](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.16949673-blue)](https://doi.org/10.5281/zenodo.16949673)

Official implementation of **scGEN**, described in:

> **scGEN: Leveraging an Adaptive Weighting Strategy for Challenging Cell Clustering in Single-Cell RNA-seq.**
> Huang F, Hu D, Dai Y, *et al.*, Hur J, Cui X, Guo K.
> *Advanced Science* (2026): e77746. [https://doi.org/10.1002/advs.77746](https://doi.org/10.1002/advs.77746)

## Introduction

Recent advancements in single-cell RNA sequencing have greatly enhanced our ability to dissect cellular heterogeneity. However, unsupervised clustering often struggles to identify transitional or developmental boundary cells, as existing methods rely on highly variable genes without considering expression levels, thereby overlooking subtle but crucial signals.

To address this challenge, we developed **scGEN** (single-cell Gene-aware Embedded Network), which captures complex cellular relationships among cells. scGEN employs adaptive feature weighting and iterative fine-tuning to prioritize ambiguous or transitional cells with overlapping transcriptional profiles.

### Key Features
- Adaptive feature weighting for better cell type identification
- Iterative fine-tuning to capture transitional cell states
- Superior performance on ambiguous cell classification
- Enhanced detection of subtle biological differences

### Performance
Evaluation across eight distinct scRNA-seq datasets demonstrated that scGEN consistently outperformed nine leading clustering approaches. Additionally, scGEN refined the classification of ~10% ambiguous cells and uncovered biologically significant differences, providing a more comprehensive view of cellular heterogeneity in the human fetal pituitary than existing methods. See the [paper](https://doi.org/10.1002/advs.77746) for full benchmark results.

## Repository Contents

```
scGEN/
├── data/               # Benchmark datasets in .mat format (8 datasets)
└── scGEN/
    ├── hvgs2csv.py     # Select the top 2,000 highly variable genes
    ├── train.py        # Main training / evaluation script
    ├── model.py        # Network definition
    ├── loss.py         # Regularized ZINB loss
    ├── opt.py          # Command-line arguments and defaults
    ├── setup.py        # Run-time settings (device, dimensions)
    └── utils.py        # Graph construction, filtering, evaluation
```

## Data and Code Availability

The [Zenodo record](https://doi.org/10.5281/zenodo.16949673) (DOI: 10.5281/zenodo.16949673) contains the benchmark datasets, the scripts used in the study, and two files needed to run this repository that are not yet included here:

- `kmeans_gpu.py`: GPU k-means used by `utils.py`. Download it and place it in the `scGEN/scGEN/` folder.
- `csv2mat.m`: MATLAB script that converts a CSV expression matrix to the `.mat` input format.

The human fetal pituitary data analyzed in the study are available from GEO under accession [GSE142653](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE142653).

## Installation

### Requirements
- Python 3 with a CUDA-capable GPU (the device is set to `cuda:0` in `scGEN/setup.py`)
- PyTorch
- scanpy, scikit-learn, scipy, numpy, pandas, munkres, tqdm
- MATLAB (only needed to convert your own data with `csv2mat.m`)

```bash
git clone https://github.com/hurlab/scGEN.git
cd scGEN
pip install torch scanpy scikit-learn scipy numpy pandas munkres tqdm
```

Then download `kmeans_gpu.py` from [Zenodo](https://doi.org/10.5281/zenodo.16949673) into `scGEN/scGEN/`.

## Input Format

scGEN reads a MATLAB `.mat` file containing two variables:

| Variable | Content |
|----------|---------|
| `X` | Expression matrix, cells × genes (normalized, top 2,000 HVGs) |
| `Y` | Cell-type labels, one per cell (used for evaluation and to set the number of clusters) |

The eight benchmark datasets in `data/` are already in this format.

## Usage

### Running on the benchmark datasets

1. **Choose the dataset(s).** Datasets are listed at the top of `scGEN/train.py`. Uncomment the dataset name(s) you want to run (e.g. `"pbmc3k"`) and comment out the others.

2. **Run from the repository root**, so that `./data/` resolves correctly, and set the hyperparameters for that dataset (see the table below):
   ```bash
   python3 scGEN/train.py --gamma 4 --alpha 0.1
   ```

Other useful options: `--runs` (number of random seeds, default 20), `--epochs` (default 400), `--lr` (default 1e-3). See `scGEN/opt.py` for the full list.

### Running on your own data

1. **Select HVGs.** Put log-normalized expression CSV files (cells as rows, genes as columns) in a folder named `dataset_norm/` and run `python3 scGEN/hvgs2csv.py`. The top 2,000 highly variable genes are written to `dataset_H/`.
2. **Create the `.mat` file.** Use `csv2mat.m` (from Zenodo) in MATLAB to build a `.mat` file with variables `X` and `Y` (see [Input Format](#input-format)).
3. **Place the data.** Copy the `.mat` file to `data/` and add its name (without the extension) to the dataset list in `scGEN/train.py`.
4. **Run scGEN** as described above.

## Hyperparameter Configuration

scGEN utilizes two key hyperparameters:
- **α** (`--alpha`): Balances the contributions of the Regularized ZINB loss and the structure-guided hard-sample contrastive loss functions
- **γ** (`--gamma`): Adjusts the attention weight assigned to hard samples in the learning process

### Best-performing Parameters by Dataset

Based on extensive parameter sensitivity analyses (α: 0.01-100, γ: 1-5), the optimal parameters for benchmark datasets are:

| Dataset      | γ (gamma) | α (alpha) |
|--------------|-----------|-----------|
| Bell         | 1         | 1         |
| hrvatin_B1   | 1         | 1         |
| hrvatin_B2   | 1         | 1         |
| pbmc3k       | 4         | 0.1       |
| Savas        | 4         | 0.1       |
| Scala        | 2         | 1         |
| Schwalbe     | 4         | 100       |
| zhang        | 4         | 10        |

### Parameter Tuning Guidelines

1. **Start with default parameters**: α=1, γ=1
2. **If results are unsatisfactory**:
   - Adjust γ for better hard-sample mining
   - Modify α based on dataset complexity

## Output and Results

The output file `result.csv` (written to the directory you run from) contains performance metrics (ACC, NMI, ARI, and F1 values) for each dataset across 20 runs, including the top two best-performing seeds with their average and standard deviation values. Results are appended, so remove or rename the file between experiments.

## Citation

If you use scGEN in your research, please cite:

> Huang F, Hu D, Dai Y, Geng A, Guan R, Quan H, McGregor BA, Wang Z, Khan N, Zhang H, Zhang J, Xia M, Gu X, Che D, Hur J, Cui X, Guo K. scGEN: Leveraging an Adaptive Weighting Strategy for Challenging Cell Clustering in Single-Cell RNA-seq. *Advanced Science*. 2026:e77746. doi:[10.1002/advs.77746](https://doi.org/10.1002/advs.77746)

```bibtex
@article{huang2026scgen,
  title   = {scGEN: Leveraging an Adaptive Weighting Strategy for Challenging Cell Clustering in Single-Cell RNA-seq},
  author  = {Huang, Fang and Hu, Dayu and Dai, Yaqi and Geng, Aoyi and Guan, Renxiang and Quan, Hao and McGregor, Brett A. and Wang, Zhihan and Khan, Nadeem and Zhang, Hongsheng and Zhang, Jinghua and Xia, Mingyang and Gu, Xiaoqiong and Che, Di and Hur, Junguk and Cui, Xiaoyu and Guo, Kai},
  journal = {Advanced Science},
  year    = {2026},
  pages   = {e77746},
  doi     = {10.1002/advs.77746}
}
```

## Contact

For questions or issues, please contact guokai8@gmail.com or open an issue on GitHub.

## Acknowledgments
The study was partially supported by the United States National Institute of Diabetes and Digestive and Kidney Diseases (R01DK130913 to Junguk Hur), the Computational Data Analysis Core of the University of North Dakota (supported by the National Institute of General Medical Sciences award P20GM113123), and the Science and Technology Research Program of Chongqing Municipal Education Commission (KJQN202200479), the Natural Science Foundation of Chongqing (CSTB2022NSCQ-LZX0033), Chongqing Medical University Program for Youth Innovation in Future Medicine (W0158), the National Natural Science Foundation of China (82200592).

# scGEN: Single-Cell Gene-Aware Embedded Network

## Introduction

Recent advancements in single-cell RNA sequencing have greatly enhanced our ability to dissect cellular heterogeneity. However, unsupervised clustering often struggles to identify transitional or developmental boundary cells, as existing methods rely on highly variable genes without considering expression levels, thereby overlooking subtle but crucial signals.

To address this challenge, we developed **scGEN** (single-cell Gene-aware Embedded Network), which captures complex cellular relationships among cells. scGEN employs adaptive feature weighting and iterative fine-tuning to prioritize ambiguous or transitional cells with overlapping transcriptional profiles. 

### Key Features
- Adaptive feature weighting for better cell type identification
- Iterative fine-tuning to capture transitional cell states
- Superior performance on ambiguous cell classification
- Enhanced detection of subtle biological differences

### Performance
Evaluation across eight distinct scRNA-seq datasets demonstrated that scGEN consistently outperformed nine leading clustering approaches. Additionally, scGEN refined the classification of ~10% ambiguous cells and uncovered biologically significant differences, providing a more comprehensive view of cellular heterogeneity in the human fetal pituitary than existing methods.

## Installation

```bash
git clone https://github.com/hurlab/scGEN.git
cd scGEN
```

## Data Preparation

scGEN accepts input data in `.mat` (MATLAB) format. You can convert your data to the required format using the provided `sv2mat.m` script in MATLAB.

## Usage

### Step-by-step workflow:

1. **Select HVGs**: Use the `hvgs2csv.py` file in the scGEN directory to filter the normalized data with top 2000 highly variable genes.

2. **Create .mat file**: Use the `csv2mat.m` file in the scGEN directory to create a `.mat` file in MATLAB.

3. **Place your data**: Put your `.mat` file in the `dataset` folder under the scGEN directory.

4. **Run scGEN**: Execute the main training script:
   ```bash
   python3 train.py
   ```

### Data Download
You can download example datasets and scripts from: [https://zenodo.org/uploads/16945598](https://zenodo.org/records/16949673)

## Hyperparameter Configuration

scGEN utilizes two key hyperparameters:
- **α**: Balances the contributions of the Regularized ZINB loss and the structure-guided hard-sample contrastive loss functions
- **γ**: Adjusts the attention weight assigned to hard samples in the learning process

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

The output file `result.csv` contains performance metrics (ACC, NMI, ARI, and F1 values) for each dataset across 20 runs, including the top two best-performing seeds with their average and standard deviation values.

## Contact

For questions or issues, please contact guokai8@gmail.com or open an issue on GitHub.

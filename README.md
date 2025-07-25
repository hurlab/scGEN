# scGEN: Leveraging Adaptive Weighting Strategy for Challenging Cell Clustering in Single-cell RNA-seq

# Introduction

Recent advancements in single-cell RNA sequencing have greatly enhanced our ability to dissect cellular heterogeneity. However, unsupervised clustering often struggles to identify transitional or developmental boundary cells, as existing methods rely on highly variable genes without considering expression levels, thereby overlooking subtle but crucial signals. To address this challenge, we developed a single-cell gene-aware embedded network (scGEN), which captures complex cellular relationships among cells. scGEN employs adaptive feature weighting and iterative fine-tuning to prioritize ambiguous or transitional cells with overlapping transcriptional profiles. Evaluation across eight distinct scRNA-seq datasets demonstrated that scGEN consistently outperformed seven leading clustering approaches. Additionally, scGEN refined the classification of ~10% ambiguous cells and uncovered biologically significant differences, providing a more comprehensive view of cellular heterogeneity in the human fetal pituitary than existing methods. These findings highlight scGEN’s ability to enhance cell-type assignments and detect subtle biological differences.

## The best-performing parameters for each dataset

| Datasets     | gamma | alpha |
|--------------|--------|--------|
| Bell         | 1      | 1      |
| hrvatin_B1   | 1      | 1      |
| hrvatin_B2   | 1      | 1      |
| pbmc3k       | 4      | 0.1    |
| Savas        | 4      | 0.1    |
| Scala        | 2      | 1      |
| Schwalbe     | 4      | 100    |
| zhang        | 4      | 10     |
|--------------|--------|--------|

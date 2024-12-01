## Title
**Exploring the Efficacy of Supervised and Self-Supervised Learning for Wildlife Image Classification: A Case Study in Tiputini, Ecuador**

## Author
Diego Villacreses  
**Last Updated:** December 1, 2024

## Overview
This repository contains the implementation and analysis code for the master's thesis focused on comparing supervised and self-supervised learning approaches for wildlife image classification. The study uses data from the Tiputini Biodiversity Station in Ecuador to evaluate model efficacy in real-world complex computer vision classification problems.

## Features
- Implementation of supervised learning models (CNNs, ViT).
- Self-supervised learning using SimCLR.
- Data preprocessing and augmentation pipelines.
- Statistical analysis of model performance.
- Visualization of results (t-SNE plots, graphs, tables).

## Repository Structure
```
.
├── main.ipynb          # Main notebook for experimentation
├── src/                # Source code directory
│   ├── config.py       # Configuration variables
│   ├── utils.py        # Utility functions
│   ├── data_processing # Data loading and preprocessing
│   ├── datasets        # Custom dataset implementations
│   ├── models          # Model architectures and training scripts
│   ├── training        # Training and evaluation functions
├── outputs/            # Directory for saving results (graphs, tables, etc.)
```

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DiegoDVillacreses/mmia_thesis.git
   cd mmia_thesis
   ```
2. Ensure the following directories are set up:
   - `data/`: Contains labeled and unlabeled datasets.
   - `outputs/`: Used for saving results like graphs and tables.

## Usage
1. Update paths and parameters in `src/config.py`.
2. Run the notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
3. Modify the global variables in the notebook to control experiments:
   - `TRAIN_GRID_SUPERVISED`: Toggle grid search for supervised training.
   - `TRAIN_ENCODER_SIMCLR`: Train the encoder using SimCLR.
   - `TRAIN_GRID_SIMCLR_CLASSIFIER`: Grid search for SimCLR classifier.
   - `COMPUTE_STATISTICAL_COMPARISON`: Perform statistical tests on model performance.

## Results
Results and outputs will be saved in the `outputs/` directory. Key outputs include:
- Accuracy and loss metrics.
- Statistical comparison between models.
- Visualizations like t-SNE and classification plots.

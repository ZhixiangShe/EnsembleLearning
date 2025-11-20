# An Ensemble Learning Framework for Pollutant Reactivity Prediction

This repository contains the official Python implementation for the paper: **"An Ensemble Learning Framework Utilizing Fusion Molecular Fingerprints for Pollutant Removal Prediction in Advanced Treatments"**.

This project presents a hierarchical ensemble learning framework that integrates multi-dimensional molecular fingerprints to accurately predict the degradation kinetics of organic micropollutants (OMPs) in advanced water treatment processes, such as ozonation and zero-valent iron (ZVI) reduction.

## Framework Overview

Our framework is designed to overcome the limitations of single-fingerprint QSAR models by synergistically fusing complementary chemical information. The architecture consists of two main layers:

1.  **Base Learners**: A series of specialized XGBoost models, each trained on a distinct molecular fingerprint (e.g., ECFP, MQN, E3FP). Each base learner becomes an expert in capturing structure-reactivity relationships from a specific dimension of chemical information—spanning compositional, topological, and conformational features.
2.  **Meta-Learner**: An optimized XGBoost model that integrates the predictions (meta-features) from the top-performing, complementary base learners with key environmental parameters (e.g., pH, temperature, reactant concentration). This allows the model to learn the complex interplay between molecular structure and process conditions.

The framework is coupled with a multi-faceted interpretability analysis using SHapley Additive exPlanations (SHAP) to elucidate the model's decision-making logic, from the influence of environmental variables down to the identification of specific atomic hotspots responsible for reactivity.

*You can add a link to a key figure from your paper here. For example:*
<!-- ![Framework Schematic](assets/figure1.png) -->


## Key Features

-   **Multi-Dimensional Fingerprint Fusion**: Supports a comprehensive suite of 15+ molecular fingerprints from `scikit-fingerprints` (skfp), including ECFP, MACCS, MQN, and E3FP.
-   **Hierarchical Ensemble Architecture**: Utilizes a stacking ensemble with XGBoost base learners and a choice of meta-learners (e.g., XGBoost, RandomForest, Ridge).
-   **Automated Hyperparameter Tuning**: Implements Bayesian optimization (`hyperopt`) for robust tuning of all models.
-   **Advanced Model Interpretation**: Provides deep mechanistic insights through SHAP analysis at both the ensemble and base-learner levels.
-   **Ablation Studies**: Quantifies the contribution of each molecular fingerprint and environmental variable to the final prediction.
-   **Inverse Design & Optimization**: Includes functionality to predict optimal environmental conditions for maximizing pollutant degradation.

## Installation

### System Requirements
-   Python 3.9+
-   Conda (Recommended for managing RDKit dependencies)

### Setup and Dependencies

It is highly recommended to create a dedicated Conda environment to ensure all dependencies are handled correctly.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ZhixiangShe/EnsembleLearning.git
    cd EnsembleLearning
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda create -n qsar_env python=3.9
    conda activate qsar_env
    ```

3.  **Install dependencies from `requirements.txt`:**
    The `requirements.txt` file contains all necessary Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: RDKit is included in the requirements file. If you encounter issues, it can also be installed via Conda: `conda install -c conda-forge rdkit`.*

## How to Run

1.  **Prepare Data**: Place your dataset (e.g., `O3 all new.xlsx` or `FeS.csv`) in the `data/` directory. Ensure the file contains a `SMILES` column and a target column (e.g., `Kob`).

2.  **Configure Script**: Open the main script (e.g., `analysis.py`) and set the `file_name` variable to match your dataset.
    ```python
    # CHOOSE YOUR DATASET HERE
    file_name = 'O3 all new'  # Options: 'FeS', 'O3 all new', 'biooxidation', etc.
    ```

3.  **Execute the framework**: Run the script from the terminal.
    ```bash
    python analysis.py
    ```

The script will execute the complete workflow: data preprocessing, base model training, ensemble model construction, evaluation, and interpretation analysis. All results, trained models, and SHAP plots will be saved to a timestamped output directory (e.g., `Ensemble_Models_Results_YYYYMMDD_HHMMSS/`).

## Citation

If you use this code or framework in your research, please cite our paper:

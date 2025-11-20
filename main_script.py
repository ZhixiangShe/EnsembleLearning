# %%
import sys

print(sys.executable)
print(sys.prefix)  # 通常指向环境的根目录
# %% md
## Import
# %%
import os
import numpy as np
from numpy.random import seed, RandomState
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from rdkit import Chem

# RDKit AllChem is used by skfp internally or for conformer generation if not using skfp's generator
try:
    from rdkit.Chem import AllChem, Draw

    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit could not be imported. Some visualization features will be unavailable.")
    RDKIT_AVAILABLE = False

# --- Model Imports ---
import xgboost as xgb
from xgboost import XGBRegressor
# MODIFICATION: Added models for Stacking ensemble
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # 导入用于处理缺失值的SimpleImputer

# MODIFICATION: Added space_eval to imports
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

# skfp imports
from skfp.fingerprints import (
    MQNsFingerprint,
    PhysiochemicalPropertiesFingerprint,
    MACCSFingerprint,
    KlekotaRothFingerprint,
    FunctionalGroupsFingerprint,
    LingoFingerprint,
    PatternFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    ECFPFingerprint,
    RDKitFingerprint,
    AutocorrFingerprint,
    E3FPFingerprint,
    USRFingerprint,
    MORSEFingerprint,
    RDFFingerprint,
    ElectroShapeFingerprint,
)
from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator

# %% md
## Configuration
# %%
# =============================================================================
# CHOOSE YOUR DATASET HERE
# Options: 'FeS', 'O3 all new', 'biooxidation', 'bioreduction'
# =============================================================================
file_name = 'O3 all new'
# file_name = 'bioreduction'
# file_name = 'FeS'
# file_name = 'O3 all new'


# =============================================================================
# General Settings
# =============================================================================
save_time = str(time.strftime('%Y%m%d_%H%M%S'))  # More unique save time
num_splits = 5  # KFold splits
hyopt_times = 10  # Bayesian optimization evaluations, reduce for faster testing e.g., 10

# Create paths for saving results
base_results_path = f'./Ensemble_Models_Results_{file_name}_{save_time}'
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

# Path for base models (one XGB per fingerprint)
base_models_path = os.path.join(base_results_path, 'Base_Models_XGB')
if not os.path.exists(base_models_path):
    os.makedirs(base_models_path)

# Path for final ensemble models
ensemble_models_path = os.path.join(base_results_path, 'Ensemble_Models')
if not os.path.exists(ensemble_models_path):
    os.makedirs(ensemble_models_path)

# %% md
## Read Data
# %%
# =============================================================================
# =============================================================================
# ==                                                                        ==
# ==                       MODIFIED CODE BLOCK                              ==
# ==        This section has been rewritten for clarity and robustness       ==
# ==        to handle different data file formats.                         ==
# ==                                                                        ==
# =============================================================================
# =============================================================================
print("\n--- Reading Data ---")

if file_name == 'FeS':
    print(f"Loading '{file_name}.csv', expecting multiple environmental features.")
    data = pd.read_csv(f'./{file_name}.csv')
    # Ensure standard SMILES column name
    if 'SMILES' not in data.columns:
        raise ValueError(f"Error: '{file_name}.csv' must contain a 'SMILES' column.")

    SMILES_original = data['SMILES'].to_numpy()

    # Extract environmental features
    other_features_names = ['pH', 'Temp', 'Pollutant Con', 'S-Fe mol', 'S-nZVI Con']
    other_features_original = data[other_features_names].to_numpy()

    # Extract target variable
    K_original = data['Kob'].to_numpy()

elif file_name == 'O3 all new':
    print(f"Loading '{file_name}.xlsx', expecting multiple environmental features.")
    data = pd.read_excel(f'./{file_name}.xlsx')
    # Ensure standard SMILES column name
    if 'SMILES' not in data.columns:
        raise ValueError(f"Error: '{file_name}.xlsx' must contain a 'SMILES' column.")

    SMILES_original = data['SMILES'].to_numpy()

    # Extract environmental features
    other_features_names = ['pH', 'Temp', 'Pollutant Con', 'O3']
    other_features_original = data[other_features_names].to_numpy()

    # Extract target variable
    K_original = data['Kob'].to_numpy()

elif file_name in ['biooxidation', 'bioreduction']:
    print(f"Loading simple dataset: '{file_name}.xlsx'. Expecting 'SMILE' and 'Kob' columns only.")
    data = pd.read_excel(f'./{file_name}.xlsx')

    # Check for required columns, case-sensitive
    if 'SMILE' not in data.columns or 'Kob' not in data.columns:
        raise ValueError(f"Error: The file '{file_name}.xlsx' must contain 'SMILE' and 'Kob' columns.")

    # To be consistent with the rest of the script, rename 'SMILE' to 'SMILES'
    # This step is important for compatibility with downstream code.
    data = data.rename(columns={'SMILE': 'SMILES'})

    # Input features
    SMILES_original = data['SMILES'].to_numpy()

    # For these datasets, there are no other environmental features.
    # Create an empty array with the correct number of rows but zero columns.
    other_features_original = np.empty((len(data), 0))
    other_features_names = []

    # Output feature
    K_original = data['Kob'].to_numpy()

else:
    raise ValueError(f"Unsupported 'file_name': {file_name}. Please add a data loading block for this file.")

print(f"\nData loading complete for '{file_name}'.")
print(f"  - Number of records loaded: {len(SMILES_original)}")
print(f"  - Number of environmental features: {len(other_features_names)}")
if other_features_names:
    print(f"  - Environmental feature names: {other_features_names}")
else:
    print("  - No environmental features will be used in the model.")
# =============================================================================
# =============================================================================
# ==                                                                        ==
# ==                          END OF MODIFIED BLOCK                         ==
# ==                                                                        ==
# =============================================================================
# =============================================================================
# %% md
## Prepare RDKit Mol Objects and Conformers (once)
# %%
## Prepare RDKit Mol Objects and Conformers (once)
print("\n--- Preparing RDKit Molecule Objects ---")
mol_transformer = MolFromSmilesTransformer()
mols_list_initial = mol_transformer.transform(SMILES_original)

# Filter out molecules for which RDKit Mol object creation failed
valid_indices = [i for i, mol in enumerate(mols_list_initial) if mol is not None]
if len(valid_indices) < len(SMILES_original):
    print(
        f"Warning: {len(SMILES_original) - len(valid_indices)} SMILES were invalid and have been removed from the dataset.")

# Apply the filter to all our data arrays
SMILES = SMILES_original[valid_indices]
mols_2d = [mols_list_initial[i] for i in valid_indices]  # For 2D fingerprints expecting Mol objects
other_features = other_features_original[valid_indices]
K = K_original[valid_indices]

if not mols_2d:
    print("FATAL ERROR: No valid molecules remained after SMILES parsing. Exiting.")
    sys.exit()

# Generate conformers - this can be time-consuming
print("\nGenerating conformers for 3D fingerprints...")
conformer_generator = ConformerGenerator(num_conformers=1, random_state=42)
mols_3d_with_conf_id = conformer_generator.transform(mols_2d)

mols_actually_with_conformers = []
indices_with_conformers_relative_to_mols2d = []  # Indices relative to mols_2d list
for i, mol_conf in enumerate(mols_3d_with_conf_id):
    if mol_conf is not None and mol_conf.GetNumConformers() > 0 and mol_conf.HasProp("conf_id"):
        mols_actually_with_conformers.append(mol_conf)
        indices_with_conformers_relative_to_mols2d.append(i)

indices_with_conformers_relative_to_mols2d = np.array(indices_with_conformers_relative_to_mols2d, dtype=int)

if len(mols_actually_with_conformers) == 0:
    print("Warning: No molecules successfully generated conformers. 3D fingerprints will be skipped.")
else:
    print(
        f"Successfully generated conformers for {len(mols_actually_with_conformers)} out of {len(mols_2d)} valid molecules.")

# %% md
## Define Fingerprint Calculators
# %%
fingerprint_calculators_config = [
    (MQNsFingerprint(), "MQNs", False),
    (PhysiochemicalPropertiesFingerprint(), "PhysiochemProps", False),
    (MACCSFingerprint(), "MACCS", False),
    # (KlekotaRothFingerprint(count=False), "KlekotaRoth_bit", False),
    (FunctionalGroupsFingerprint(count=False), "FuncGroups_bit", False),
    # (LingoFingerprint(), "Lingo", False),
    (PatternFingerprint(), "PatternFP", False),
    (AtomPairFingerprint(use_3D=False), "AtomPair_2D", False),
    (TopologicalTorsionFingerprint(), "TopologicalTorsion", False),
    (ECFPFingerprint(), "ECFP", False),
    (RDKitFingerprint(), "RDKitFP", False),
    (AutocorrFingerprint(use_3D=False), "Autocorr_2D", False),
    (E3FPFingerprint(random_state=42), "E3FP", True),
    (USRFingerprint(errors="ignore"), "USR", True),
    (MORSEFingerprint(), "MORSE", True),
    (RDFFingerprint(), "RDF", True),
    (ElectroShapeFingerprint(errors="ignore"), "ElectroShape", True),
]
# %% md
## Define XGBoost Hyperparameter Space for Base Models
# %%
# We only need the XGBoost configuration for the base models
xgb_config = {
    'model': XGBRegressor,
    'params': {
        'random_state': 190425,
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50,
    },
    'is_xgb': True,
    'bayes_space': {
        'n_estimators': hp.quniform('n_estimators', 100, 1500, 50),
        'max_depth': hp.quniform('max_depth', 2, 15, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    }
}

# The rest of the script from this point onwards should work correctly
# with the data structures created in the modified 'Read Data' section.
# The logic for handling empty 'other_features' arrays is already in place.
# ... (The rest of your code from PHASE 1 onwards remains unchanged) ...

# %% md
## PHASE 1: Train Base Models (XGBoost per Fingerprint) and Generate Out-of-Fold (OOF) Predictions
# %%
## PHASE 1: Train Base Models (XGBoost per Fingerprint) and Generate Out-of-Fold (OOF) Predictions
print("\n" + "=" * 80)
print("PHASE 1: Training Base XGBoost Models for Each Fingerprint")
print("=" * 80 + "\n")

# Dictionaries to store results from Phase 1
base_model_oof_predictions = {}
base_models_per_fold = {}
base_model_performance = []

# MODIFICATION: Dictionary to store final models, scalers, and imputers for inverse design
final_base_models_and_processors = {}

# --- This loop is a modified version of the original main loop ---
for fp_calculator_instance, fp_name, needs_conformers in fingerprint_calculators_config:
    print(f"\n{'=' * 60}\nProcessing Fingerprint for Base Model: {fp_name}\n{'=' * 60}")

    # --- Step 1 & 2: Calculate Fingerprint and Cross-Validation ---
    current_fp_path = os.path.join(base_models_path, fp_name)
    if not os.path.exists(current_fp_path):
        os.makedirs(current_fp_path)

    Y_for_model = K
    # MODIFICATION: Environmental variables are not used in the base models, so we don't need to subset them here.
    # current_other_features_for_fp = other_features
    input_indices = np.arange(len(K))

    if needs_conformers:
        if len(mols_actually_with_conformers) == 0:
            print(f"Skipping {fp_name} because it requires conformers and none were generated.")
            continue
        input_mols_for_fp = mols_actually_with_conformers
        Y_for_model = K[indices_with_conformers_relative_to_mols2d]
        # MODIFICATION: No longer need to subset env features here.
        # current_other_features_for_fp = other_features[indices_with_conformers_relative_to_mols2d]
        input_indices = indices_with_conformers_relative_to_mols2d
    elif isinstance(fp_calculator_instance, LingoFingerprint):
        input_mols_for_fp = SMILES
    else:
        input_mols_for_fp = mols_2d

    if len(input_mols_for_fp) == 0:
        print(f"Skipping {fp_name} due to no valid input molecules.")
        continue

    try:
        print(f"Calculating {fp_name} fingerprints...")
        FP_raw = fp_calculator_instance.transform(input_mols_for_fp)
        FP = np.asarray(FP_raw.toarray() if hasattr(FP_raw, "toarray") else FP_raw)
    except Exception as e:
        print(f"Error calculating {fp_name}: {e}. Skipping.")
        continue

    # MODIFICATION: The feature for the base model is ONLY the fingerprint.
    # Environmental variables will be added later in the ensemble stage.
    feature_w_fp_unscaled = FP

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=12343)
    oof_preds_for_fp = np.zeros(len(K))
    fold_models = []
    y_test_all_folds, y_pred_all_folds = [], []

    for fold_num, (Train_index_local, Test_index_local) in enumerate(kf.split(feature_w_fp_unscaled, Y_for_model), 1):
        X_train_unscaled, X_test_unscaled = feature_w_fp_unscaled[Train_index_local], feature_w_fp_unscaled[
            Test_index_local]
        y_train, y_test = Y_for_model[Train_index_local], Y_for_model[Test_index_local]
        scaler = StandardScaler().fit(X_train_unscaled)
        imputer = SimpleImputer(strategy='mean').fit(scaler.transform(X_train_unscaled))
        X_train = imputer.transform(scaler.transform(X_train_unscaled))
        X_test = imputer.transform(scaler.transform(X_test_unscaled))


        # Hyperopt
        def model_cv_bayes(params_b):
            params_b['n_estimators'] = int(params_b['n_estimators'])
            params_b['max_depth'] = int(params_b['max_depth'])
            static_params = xgb_config.get('params', {}).copy()
            if 'early_stopping_rounds' in static_params: del static_params['early_stopping_rounds']
            tmp_model = xgb_config['model'](**params_b, **static_params)
            inner_kf = KFold(n_splits=3, shuffle=True, random_state=fold_num)
            scores = cross_val_score(tmp_model, X_train, y_train, scoring='neg_mean_squared_error', cv=inner_kf)
            return {'loss': -np.mean(scores), 'status': STATUS_OK}


        trials_bayes = Trials()
        best_bayes_params = fmin(model_cv_bayes, space=xgb_config['bayes_space'], algo=tpe.suggest,
                                 max_evals=hyopt_times, trials=trials_bayes,
                                 rstate=np.random.default_rng(fold_num * 10))
        final_model_params = space_eval(xgb_config['bayes_space'], best_bayes_params)
        final_model_params['n_estimators'] = int(final_model_params['n_estimators'])
        final_model_params['max_depth'] = int(final_model_params['max_depth'])

        model = xgb_config['model'](**final_model_params, **xgb_config.get('params', {}))
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)

        global_test_indices = input_indices[Test_index_local]
        oof_preds_for_fp[global_test_indices] = y_pred
        fold_models.append(model)
        y_test_all_folds.append(y_test)
        y_pred_all_folds.append(y_pred)

    # --- After all folds ---
    base_model_oof_predictions[fp_name] = oof_preds_for_fp
    base_models_per_fold[fp_name] = fold_models
    y_test_agg = np.concatenate(y_test_all_folds)
    y_pred_agg = np.concatenate(y_pred_all_folds)
    r2 = r2_score(y_test_agg, y_pred_agg)
    mae = mean_absolute_error(y_test_agg, y_pred_agg)
    print(f"\nPerformance for base model XGBoost + {fp_name}: R2 = {r2:.4f}, MAE = {mae:.4f}")
    base_model_performance.append({'fingerprint': fp_name, 'model': 'XGBoost', 'R2_total': r2, 'MAE_total': mae})

    # MODIFICATION: Train and store a final model on ALL data for this fingerprint
    print(f"Training final base model for {fp_name} on all available data...")
    final_scaler = StandardScaler().fit(feature_w_fp_unscaled)
    final_imputer = SimpleImputer(strategy='mean').fit(final_scaler.transform(feature_w_fp_unscaled))
    X_final_train = final_imputer.transform(final_scaler.transform(feature_w_fp_unscaled))


    # Re-run hyperopt for best performance on the full dataset
    def final_model_cv_bayes(params_b):
        params_b['n_estimators'] = int(params_b['n_estimators'])
        params_b['max_depth'] = int(params_b['max_depth'])
        static_params = xgb_config.get('params', {}).copy()
        if 'early_stopping_rounds' in static_params: del static_params['early_stopping_rounds']
        tmp_model = xgb_config['model'](**params_b, **static_params)
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(tmp_model, X_final_train, Y_for_model, scoring='neg_mean_squared_error', cv=inner_kf)
        return {'loss': -np.mean(scores), 'status': STATUS_OK}


    final_trials = Trials()
    best_final_params = fmin(final_model_cv_bayes, space=xgb_config['bayes_space'], algo=tpe.suggest,
                             max_evals=hyopt_times, trials=final_trials, rstate=np.random.default_rng(42))
    final_params = space_eval(xgb_config['bayes_space'], best_final_params)
    final_params['n_estimators'] = int(final_params['n_estimators'])
    final_params['max_depth'] = int(final_params['max_depth'])

    final_base_model = xgb_config['model'](**final_params, **xgb_config.get('params', {}))
    # No early stopping for the final model, train on all data
    if 'early_stopping_rounds' in final_base_model.get_params():
        final_base_model.fit(X_final_train, Y_for_model, eval_set=[(X_final_train, Y_for_model)], verbose=False)
    else:
        final_base_model.fit(X_final_train, Y_for_model)

    final_base_models_and_processors[fp_name] = {
        'model': final_base_model,
        'scaler': final_scaler,
        'imputer': final_imputer,
        'fp_calculator': fp_calculator_instance,
        'needs_conformers': needs_conformers
    }
    print(f"Final model for {fp_name} is trained and stored.")

# Save summary of all base models
base_model_summary_df = pd.DataFrame(base_model_performance)
base_model_summary_df.to_csv(os.path.join(base_models_path, 'base_models_summary.csv'), index=False)
print("\n--- Base Model Performance Summary ---")
print(base_model_summary_df)
# (The rest of the script continues from here without changes)

# The following phases (2, 2.5, 3, 3.5, 4, 5, 6, 7) are already designed
# to handle the case where `other_features` is empty and will work correctly
# with the data loaded for 'biooxidation' and 'bioreduction'.
# ... (rest of the script) ...
# %% md
## PHASE 2: Assemble Meta-Features from OOF Predictions
# %%
## PHASE 2: Base Model Selection by Performance and Meta-Feature Assembly
print("\n" + "=" * 80)
print("PHASE 2: Selecting Top-Performing Base Models for Meta-Feature Assembly")
print("=" * 80 + "\n")

if not base_model_performance:
    print("No base models were trained successfully. Exiting.")
    sys.exit()

# =============================================================================
#  CONFIGURATION FOR SELECTION
# =============================================================================
# The total number of base models to select for the final ensemble.
# These will be the models with the highest R2 scores from Phase 1.
NUM_MODELS_TO_SELECT = 4

# =============================================================================
#  STEP 1: SELECT TOP MODELS BASED ON PERFORMANCE
# =============================================================================
# Create a DataFrame from the performance results of Phase 1
base_model_summary_df = pd.DataFrame(base_model_performance)

# NEW LOGIC: Sort the models by R2 score in descending order and take the top N
print(f"Selecting the top {NUM_MODELS_TO_SELECT} base models based purely on R2 performance...")
sorted_models_df = base_model_summary_df.sort_values(by='R2_total', ascending=False)
top_n_models_df = sorted_models_df.head(NUM_MODELS_TO_SELECT)

# Extract the names of the top-performing fingerprints
top_fp_names = top_n_models_df['fingerprint'].tolist()

# =============================================================================
#  STEP 2: ASSEMBLE FINAL META-FEATURES AND SAVE
# =============================================================================
# Now we use the performance-selected list `top_fp_names`
print(f"\nAssembling meta-features from the {len(top_fp_names)} selected models:")
print("Selected Models and their Performance:")
print(top_n_models_df[['fingerprint', 'R2_total', 'MAE_total']].to_string(index=False))

# Create the new feature matrix X_meta using only the selected models' predictions
meta_feature_names = top_fp_names
X_meta = np.column_stack([base_model_oof_predictions[fp_name] for fp_name in meta_feature_names])
Y_meta = K  # The original target variable

print(f"\nFinal meta-feature matrix (X_meta) created with shape: {X_meta.shape}")

# Saving the results for these specific models
base_preds_df = pd.DataFrame(X_meta, columns=[f'pred_{name}' for name in meta_feature_names])
base_preds_df['target'] = Y_meta
cols = ['target'] + [col for col in base_preds_df if col != 'target']
base_preds_df = base_preds_df[cols]
output_path_base_preds = os.path.join(base_models_path,
                                      f'base_models_performance_selected_top{len(top_fp_names)}_oof_predictions.csv')
base_preds_df.to_csv(output_path_base_preds, index=False)
print(f"\nSaved the performance-selected base model OOF predictions to: {output_path_base_preds}")

meta_df = pd.DataFrame(X_meta, columns=[f'pred_{name}' for name in meta_feature_names])
meta_df.to_csv(os.path.join(ensemble_models_path, f'meta_features_for_stacking_top{len(top_fp_names)}.csv'),
               index=False)

# Optional: Visualize the correlation matrix of the selected models
print("\nCalculating correlation matrix of the SELECTED top-performing models...")
selected_oof_predictions_df = pd.DataFrame(
    {fp_name: base_model_oof_predictions[fp_name] for fp_name in top_fp_names}
)
correlation_matrix_selected = selected_oof_predictions_df.corr()
plt.figure(figsize=(8, 6), dpi=100)
sns.heatmap(correlation_matrix_selected, cmap='viridis', annot=True, fmt=".2f")
plt.title(f'Correlation Matrix of Top {len(top_fp_names)} Performing Models', fontsize=16)
plt.savefig(os.path.join(base_models_path, 'selected_model_correlation_heatmap.png'))
plt.close()
print(f"Correlation heatmap for selected models saved to: {base_models_path}")

# %% md
## PHASE 2.5: Outlier Identification and Data Pruning based on Base Model Errors
# %%
## PHASE 2.5: Outlier Identification and Data Pruning based on Base Model Errors
print("\n" + "=" * 80)
print("PHASE 2.5: Identifying and Pruning High-Error Data Points")
print("=" * 80 + "\n")

# =============================================================================
#  CONFIGURATION FOR PRUNING
#  Set the percentile of data to KEEP.
#  For example, 95 means we remove the 5% of points with the highest error.
#  Set to 100 to disable pruning.
# =============================================================================
KEEP_PERCENTILE = 95  # You can experiment with this value (e.g., 90, 98, 100)

# Prepare a DataFrame of the original features for easy filtering and saving.
# Note: These arrays (SMILES, K, other_features) are already filtered for valid RDKit molecules.
# MODIFICATION: This now correctly handles the case where other_features is empty.
if other_features.shape[1] > 0:
    original_data_df = pd.DataFrame(other_features, columns=other_features_names)
else:
    original_data_df = pd.DataFrame()  # Start with an empty dataframe
original_data_df.insert(0, 'SMILES', SMILES)
original_data_df.insert(len(original_data_df.columns), 'Kob', K)

if KEEP_PERCENTILE < 100:
    # We already have the base model predictions (X_meta) and true values (Y_meta)

    # Calculate the absolute error for each base model's OOF predictions
    absolute_errors = np.abs(X_meta - Y_meta.reshape(-1, 1))

    # Calculate the mean absolute error for each data point across all base models
    mean_absolute_error_per_point = np.mean(absolute_errors, axis=1)

    # Find the error threshold based on the desired percentile
    error_threshold = np.percentile(mean_absolute_error_per_point, KEEP_PERCENTILE)

    # Identify the indices of the data points to KEEP
    indices_to_keep = np.where(mean_absolute_error_per_point <= error_threshold)[0]

    num_original_samples = len(Y_meta)
    num_pruned_samples = len(indices_to_keep)
    num_removed = num_original_samples - num_pruned_samples

    print(f"Pruning configuration: Keeping {KEEP_PERCENTILE}% of the data.")
    print(f"Error threshold (max mean absolute error to keep): {error_threshold:.4f}")
    print(f"Identified {num_removed} high-error data points to remove.")
    print(f"Dataset size reduced from {num_original_samples} to {num_pruned_samples}.")

    # Create the new, pruned datasets for Phase 3
    X_meta_pruned = X_meta[indices_to_keep, :]
    Y_meta_pruned = Y_meta[indices_to_keep]

    ## MODIFICATION: Prune the environmental features using the same indices
    other_features_pruned = other_features[indices_to_keep, :]

    # Create the pruned dataset from the original data and save to Excel
    pruned_dataset_df = original_data_df.iloc[indices_to_keep].copy()
    # Optional: Add the error score to the saved file for analysis
    pruned_dataset_df['mean_oof_error'] = mean_absolute_error_per_point[indices_to_keep]

else:
    print("Pruning is disabled (KEEP_PERCENTILE = 100). Using the full dataset for ensembles.")
    # If pruning is disabled, the "pruned" dataset is just the original dataset
    X_meta_pruned = X_meta
    Y_meta_pruned = Y_meta

    ## MODIFICATION: If not pruning, the "pruned" environmental features are the original ones
    other_features_pruned = other_features

    # If not pruning, the "pruned dataset" is the full original dataset
    pruned_dataset_df = original_data_df.copy()

# Save the final dataset used for ensembles to an Excel file
output_xlsx_path = os.path.join(ensemble_models_path, 'dataset_used_for_ensemble.xlsx')
try:
    pruned_dataset_df.to_excel(output_xlsx_path, index=False)
    print(f"\nSaved the new dataset (for ensemble training) to: {output_xlsx_path}")
except ImportError:
    print(
        "\nWarning: `openpyxl` is not installed. Cannot save to .xlsx format. Please install it using: pip install openpyxl")
    # Fallback to CSV if openpyxl is not available
    output_csv_path = os.path.join(ensemble_models_path, 'dataset_used_for_ensemble.csv')
    pruned_dataset_df.to_csv(output_csv_path, index=False)
    print(f"Saved the new dataset as CSV instead: {output_csv_path}")

# %% md
## PHASE 3: Train and Evaluate Ensemble Models
# %%
## PHASE 3: Train and Evaluate Ensemble Models (with Meta-Model Tuning and Detailed CV Export)
print("\n" + "=" * 80)
print("PHASE 3: Training, Tuning, and Evaluating Ensemble Models")
print("=" * 80 + "\n")

# =============================================================================
#  NECESSARY IMPORTS FOR THIS PHASE
# =============================================================================
from sklearn.model_selection import GridSearchCV
# NEW: Import 'clone' to create fresh model instances for each fold
from sklearn.base import clone

# =============================================================================
#  PREPARE COMBINED INPUTS
# =============================================================================
# Combine meta-features (OOF predictions) and environmental features
# THIS IS THE KEY STEP WHERE ENVIRONMENTAL VARIABLES ARE INTRODUCED
# MODIFICATION: This now correctly handles cases where other_features_pruned is empty.
X_ensemble_input = np.concatenate((X_meta_pruned, other_features_pruned), axis=1)
Y_ensemble_target = Y_meta_pruned
ensemble_feature_names = meta_feature_names + other_features_names

print(f"Created combined feature matrix for ensemble learning.")
print(f"  - Shape of final combined input (X_ensemble_input): {X_ensemble_input.shape}")

ensemble_results = []
kf_stack = KFold(n_splits=num_splits, shuffle=True, random_state=98765)

# =============================================================================
#  ENSEMBLE METHOD 1: AVERAGING (BAGGING-LIKE BASELINE) WITH DETAILED CV
# =============================================================================
print("\n--- Method 1: Averaging (Bagging-like) with Cross-Validation ---")

# Lists to store metrics and predictions from each fold
# For Test set
fold_r2_scores_avg_test = []
fold_mae_scores_avg_test = []
# For Training set
fold_r2_scores_avg_train = []
fold_mae_scores_avg_train = []

all_test_predictions_avg = []
all_train_predictions_avg = []

# Manual K-Fold cross-validation loop. Note: Averaging uses X_meta_pruned, not X_ensemble_input.
for fold_idx, (train_index, test_index) in enumerate(kf_stack.split(X_meta_pruned, Y_ensemble_target), 1):
    X_train_meta, X_test_meta = X_meta_pruned[train_index], X_meta_pruned[test_index]
    y_train, y_test = Y_ensemble_target[train_index], Y_ensemble_target[test_index]

    # Predictions
    y_pred_test_avg = np.mean(X_test_meta, axis=1)
    y_pred_train_avg = np.mean(X_train_meta, axis=1)  # Predictions on training data

    # Test set metrics
    r2_fold_test = r2_score(y_test, y_pred_test_avg)
    mae_fold_test = mean_absolute_error(y_test, y_pred_test_avg)
    fold_r2_scores_avg_test.append(r2_fold_test)
    fold_mae_scores_avg_test.append(mae_fold_test)

    # Training set metrics
    r2_fold_train = r2_score(y_train, y_pred_train_avg)
    mae_fold_train = mean_absolute_error(y_train, y_pred_train_avg)
    fold_r2_scores_avg_train.append(r2_fold_train)
    fold_mae_scores_avg_train.append(mae_fold_train)

    # Storing predictions for CSV export
    test_preds_df = pd.DataFrame({'fold': fold_idx, 'set': 'test', 'y_true': y_test, 'y_pred': y_pred_test_avg})
    all_test_predictions_avg.append(test_preds_df)
    train_preds_df = pd.DataFrame({'fold': fold_idx, 'set': 'train', 'y_true': y_train, 'y_pred': y_pred_train_avg})
    all_train_predictions_avg.append(train_preds_df)

# Test set summary
r2_mean_avg_test = np.mean(fold_r2_scores_avg_test)
r2_std_avg_test = np.std(fold_r2_scores_avg_test)
mae_mean_avg_test = np.mean(fold_mae_scores_avg_test)
mae_std_avg_test = np.std(fold_mae_scores_avg_test)

# Training set summary
r2_mean_avg_train = np.mean(fold_r2_scores_avg_train)
r2_std_avg_train = np.std(fold_r2_scores_avg_train)
mae_mean_avg_train = np.mean(fold_mae_scores_avg_train)
mae_std_avg_train = np.std(fold_mae_scores_avg_train)

print(
    f"Averaging Ensemble -> CV Test R2: {r2_mean_avg_test:.4f} ± {r2_std_avg_test:.4f}, CV Test MAE: {mae_mean_avg_test:.4f} ± {mae_std_avg_test:.4f}")
print(
    f"Averaging Ensemble -> CV Train R2: {r2_mean_avg_train:.4f} ± {r2_std_avg_train:.4f}, CV Train MAE: {mae_mean_avg_train:.4f} ± {mae_std_avg_train:.4f}")

ensemble_results.append({
    'Ensemble Type': 'Averaging (Bagging-like)',
    'R2_test_mean': r2_mean_avg_test, 'R2_test_std': r2_std_avg_test,
    'MAE_test_mean': mae_mean_avg_test, 'MAE_test_std': mae_std_avg_test,
    'R2_train_mean': r2_mean_avg_train, 'R2_train_std': r2_std_avg_train,
    'MAE_train_mean': mae_mean_avg_train, 'MAE_train_std': mae_std_avg_train
})

safe_name_avg = "Averaging_Bagging-like"
final_test_preds_avg_df = pd.concat(all_test_predictions_avg, ignore_index=True)
test_preds_output_path_avg = os.path.join(ensemble_models_path, f'predictions_test_set_{safe_name_avg}_{save_time}.csv')
final_test_preds_avg_df.to_csv(test_preds_output_path_avg, index=False)
print(f"  -> Saved test set predictions to: {test_preds_output_path_avg}")

final_train_preds_avg_df = pd.concat(all_train_predictions_avg, ignore_index=True)
train_preds_output_path_avg = os.path.join(ensemble_models_path,
                                           f'predictions_train_set_{safe_name_avg}_{save_time}.csv')
final_train_preds_avg_df.to_csv(train_preds_output_path_avg, index=False)
print(f"  -> Saved training set predictions to: {train_preds_output_path_avg}")

# =============================================================================
#  META-MODEL HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 80)
print("STARTING META-MODEL HYPERPARAMETER TUNING")
print("=" * 80 + "\n")

param_grid_rf = {
    'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 1.0]
}
param_grid_xgb = {
    'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2], 'subsample': [0.7, 1.0], 'colsample_bytree': [0.7, 1.0]
}

models_to_tune = {
    'Stacking_RandomForest': (RandomForestRegressor(random_state=42, n_jobs=-1), param_grid_rf),
    'Boosting-like_XGBoost': (XGBRegressor(random_state=42, eval_metric='rmse'), param_grid_xgb)
}

tuned_meta_models = {}
for name, (estimator, param_grid) in models_to_tune.items():
    print(f"\n--- Tuning Meta-Model: {name} ---")
    grid_search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, cv=kf_stack,
        scoring='r2', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_ensemble_input, Y_ensemble_target)  # Use full X_ensemble_input for tuning
    print(f"Best R2 score from tuning: {grid_search.best_score_:.4f}")
    print(f"Best parameters found for {name}:\n{grid_search.best_params_}")
    tuned_meta_models[name] = grid_search.best_estimator_

# RidgeCV tunes its alpha internally during fit, so we just instantiate it.
# It will be fit during the cross-validation loop below.
tuned_meta_models['Stacking_Ridge'] = RidgeCV(alphas=np.logspace(-3, 3, 10))

# =============================================================================
#  EVALUATE FINAL, TUNED META-MODELS WITH DETAILED CV
# =============================================================================
print("\n" + "=" * 80)
print("EVALUATING FINAL TUNED META-MODELS")
print("=" * 80 + "\n")

for name, meta_model_prototype in tuned_meta_models.items():
    print(f"\n-- Evaluating Final Tuned Model: {name} --")
    # For Test set
    fold_r2_scores_test, fold_mae_scores_test = [], []
    # For Training set
    fold_r2_scores_train, fold_mae_scores_train = [], []

    all_test_predictions, all_train_predictions = [], []

    for fold_idx, (train_index, test_index) in enumerate(kf_stack.split(X_ensemble_input, Y_ensemble_target), 1):
        X_train, X_test = X_ensemble_input[train_index], X_ensemble_input[test_index]
        y_train, y_test = Y_ensemble_target[train_index], Y_ensemble_target[test_index]

        meta_model = clone(meta_model_prototype)  # Clone to ensure fresh model for each fold

        # Fit the model (RidgeCV tunes alpha here)
        meta_model.fit(X_train, y_train)

        # Predictions
        y_pred_test = meta_model.predict(X_test)
        y_pred_train = meta_model.predict(X_train)  # Predictions on training data

        # Test set metrics
        r2_fold_test = r2_score(y_test, y_pred_test)
        mae_fold_test = mean_absolute_error(y_test, y_pred_test)
        fold_r2_scores_test.append(r2_fold_test)
        fold_mae_scores_test.append(mae_fold_test)

        # Training set metrics
        r2_fold_train = r2_score(y_train, y_pred_train)
        mae_fold_train = mean_absolute_error(y_train, y_pred_train)
        fold_r2_scores_train.append(r2_fold_train)
        fold_mae_scores_train.append(mae_fold_train)

        # Storing predictions for CSV export
        test_preds_df = pd.DataFrame({'fold': fold_idx, 'set': 'test', 'y_true': y_test, 'y_pred': y_pred_test})
        all_test_predictions.append(test_preds_df)
        train_preds_df = pd.DataFrame({'fold': fold_idx, 'set': 'train', 'y_true': y_train, 'y_pred': y_pred_train})
        all_train_predictions.append(train_preds_df)

    # Test set summary
    r2_mean_test, r2_std_test = np.mean(fold_r2_scores_test), np.std(fold_r2_scores_test)
    mae_mean_test, mae_std_test = np.mean(fold_mae_scores_test), np.std(fold_mae_scores_test)

    # Training set summary
    r2_mean_train, r2_std_train = np.mean(fold_r2_scores_train), np.std(fold_r2_scores_train)
    mae_mean_train, mae_std_train = np.mean(fold_mae_scores_train), np.std(fold_mae_scores_train)

    print(
        f"{name} -> CV Test R2: {r2_mean_test:.4f} ± {r2_std_test:.4f}, CV Test MAE: {mae_mean_test:.4f} ± {mae_std_test:.4f}")
    print(
        f"{name} -> CV Train R2: {r2_mean_train:.4f} ± {r2_std_train:.4f}, CV Train MAE: {mae_mean_train:.4f} ± {mae_std_train:.4f}")

    ensemble_results.append({
        'Ensemble Type': f'Tuned_{name}',
        'R2_test_mean': r2_mean_test, 'R2_test_std': r2_std_test,
        'MAE_test_mean': mae_mean_test, 'MAE_test_std': mae_std_test,
        'R2_train_mean': r2_mean_train, 'R2_train_std': r2_std_train,
        'MAE_train_mean': mae_mean_train, 'MAE_train_std': mae_std_train
    })

    safe_name = name.replace(' (', '_').replace(')', '').replace(':', '')
    final_test_preds_df = pd.concat(all_test_predictions, ignore_index=True)
    test_preds_output_path = os.path.join(ensemble_models_path, f'predictions_test_set_{safe_name}_{save_time}.csv')
    final_test_preds_df.to_csv(test_preds_output_path, index=False)
    print(f"  -> Saved test set predictions to: {test_preds_output_path}")

    final_train_preds_df = pd.concat(all_train_predictions, ignore_index=True)
    train_preds_output_path = os.path.join(ensemble_models_path, f'predictions_train_set_{safe_name}_{save_time}.csv')
    final_train_preds_df.to_csv(train_preds_output_path, index=False)
    print(f"  -> Saved training set predictions to: {train_preds_output_path}")

    # Plotting (Test set performance)
    plt.figure(figsize=(8, 8), dpi=150)
    color = 'green' if 'Ridge' in name else ('purple' if 'Forest' in name else 'red')
    plt.scatter(final_test_preds_df['y_true'], final_test_preds_df['y_pred'], alpha=0.6, s=50, c=color,
                edgecolors='black')
    min_val = min(final_test_preds_df['y_true'].min(), final_test_preds_df['y_pred'].min())
    max_val = max(final_test_preds_df['y_true'].max(), final_test_preds_df['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='black', linewidth=2)
    plt.xlabel('Kob_True [min-1]', fontsize=18)
    plt.ylabel('Kob_Predicted [min-1]', fontsize=18)
    plt.title(f'Tuned {name} Ensemble (OOS Test)\nR2={r2_mean_test:.4f}, MAE={mae_mean_test:.2f}', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(ensemble_models_path, f'ensemble_Tuned_{safe_name}_plot_test_{save_time}.png'))
    plt.close()

## Final Summary
ensemble_summary_df = pd.DataFrame(ensemble_results)
# Update column renaming to reflect train/test differentiation
ensemble_summary_df = ensemble_summary_df.rename(columns={
    'R2_test_mean': 'R2 Test (Mean)', 'R2_test_std': 'R2 Test (Std)',
    'MAE_test_mean': 'MAE Test (Mean)', 'MAE_test_std': 'MAE Test (Std)',
    'R2_train_mean': 'R2 Train (Mean)', 'R2_train_std': 'R2 Train (Std)',
    'MAE_train_mean': 'MAE Train (Mean)', 'MAE_train_std': 'MAE Train (Std)'
})
ensemble_summary_df.to_csv(os.path.join(ensemble_models_path, 'ensemble_models_summary.csv'), index=False)

print("\n" + "=" * 80)
print("FINAL ENSEMBLE PERFORMANCE SUMMARY")
print("=" * 80 + "\n")
print(ensemble_summary_df.to_string())

print(f"\nAll processing finished. Results saved in: {base_results_path}")

# Correction for downstream phases if it relies on a specific R2 column name from the summary
# For example, if PHASE 3.5 expects 'R2 (Mean)', ensure it's handled or the key is updated.
# Let's assume the downstream ablation study should use the TEST set performance.
# We can sort by 'R2 Test (Mean)' later.
final_estimators = tuned_meta_models
# %% md
## Final Summary
# %%
# This cell seems to be a duplicate or old version. The previous cell already handles this logic more completely.
# We will rely on the `ensemble_summary_df` created in the previous cell.
# ensemble_summary_df = pd.DataFrame(ensemble_results)
# ensemble_summary_df = ensemble_summary_df.rename(columns={
#     'R2_mean': 'R2 (Mean)', 'R2_std': 'R2 (Std)',
#     'MAE_mean': 'MAE (Mean)', 'MAE_std': 'MAE (Std)'
# })
# ensemble_summary_df.to_csv(os.path.join(ensemble_models_path, 'ensemble_models_summary.csv'), index=False)

print("\n" + "=" * 80)
print("FINAL ENSEMBLE PERFORMANCE SUMMARY (from previous cell)")
print("=" * 80 + "\n")
print(ensemble_summary_df.to_string())

print(f"\nAll processing finished. Results saved in: {base_results_path}")

# Correction for downstream phases
final_estimators = tuned_meta_models
# %% md
## PHASE 3.5: Detailed Ablation Study by Feature Removal (with Per-Fold Results)
# %%
## PHASE 3.5: Detailed Ablation Study by Feature Removal (with Per-Fold Results)
print("\n" + "=" * 80)
print("PHASE 3.5: Detailed Ablation Study by Feature Removal (with Per-Fold Results)")
print("=" * 80 + "\n")

# --- Step 1: Identify the best ensemble model ---
# MODIFICATION: Use the correct column name 'R2 Test (Mean)' for sorting
stacking_results = ensemble_summary_df[ensemble_summary_df['Ensemble Type'].str.contains('Tuned_')]
if stacking_results.empty:
    print("Error: No tuned stacking models found. Cannot perform detailed ablation study.")
    sys.exit()

best_model_info_row = stacking_results.loc[stacking_results['R2 Test (Mean)'].idxmax()]
best_model_name_full = best_model_info_row['Ensemble Type']
model_key = best_model_name_full.replace('Tuned_', '')
best_meta_model_prototype = tuned_meta_models[model_key]

print(f"Using '{best_model_name_full}' as the final model for the ablation study.")

# --- Step 2: Define helper function and prepare for the loop ---
ablation_results_list = []
X_full = X_ensemble_input
y_full = Y_ensemble_target
all_feature_names = ensemble_feature_names


def run_cv_and_get_fold_scores(model_proto, X, y, kf):
    r2_scores, mae_scores = [], []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = clone(model_proto)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
    return r2_scores, mae_scores


# --- Step 3: Evaluate the Baseline (Full Model) ---
print("\nEvaluating baseline (full model) to get per-fold metrics...")
baseline_r2_folds, baseline_mae_folds = run_cv_and_get_fold_scores(
    model_proto=best_meta_model_prototype, X=X_full, y=y_full, kf=kf_stack
)
baseline_r2_mean = np.mean(baseline_r2_folds)
baseline_mae_mean = np.mean(baseline_mae_folds)
print(f"Baseline Performance (Full Model) -> R2: {baseline_r2_mean:.4f}, MAE: {baseline_mae_mean:.4f}\n")

baseline_result = {
    'Removed Feature': 'None (Full Model)', 'R2 (Mean)': baseline_r2_mean,
    'R2 (Std)': np.std(baseline_r2_folds), 'MAE (Mean)': baseline_mae_mean,
    'MAE (Std)': np.std(baseline_mae_folds)
}
for i, (r2, mae) in enumerate(zip(baseline_r2_folds, baseline_mae_folds), 1):
    baseline_result[f'R2_Fold_{i}'] = r2
    baseline_result[f'MAE_Fold_{i}'] = mae
ablation_results_list.append(baseline_result)

# --- Step 4: Loop through each feature, remove it, and re-evaluate ---
print("Starting evaluation loop...")
for i, feature_name in enumerate(all_feature_names):
    print(f"  > Evaluating model without feature: '{feature_name}'...")
    X_ablated = np.delete(X_full, i, axis=1)
    r2_folds, mae_folds = run_cv_and_get_fold_scores(
        model_proto=best_meta_model_prototype, X=X_ablated, y=y_full, kf=kf_stack
    )
    r2_mean, mae_mean = np.mean(r2_folds), np.mean(mae_folds)
    print(f"    ... Result: R2={r2_mean:.4f}, MAE={mae_mean:.4f}")

    scenario_result = {
        'Removed Feature': feature_name, 'R2 (Mean)': r2_mean,
        'R2 (Std)': np.std(r2_folds), 'MAE (Mean)': mae_mean,
        'MAE (Std)': np.std(mae_folds)
    }
    for j, (r2, mae) in enumerate(zip(r2_folds, mae_folds), 1):
        scenario_result[f'R2_Fold_{j}'] = r2
        scenario_result[f'MAE_Fold_{j}'] = mae
    ablation_results_list.append(scenario_result)

# --- Step 5: Finalize, Analyze, and Display the Results ---
ablation_df = pd.DataFrame(ablation_results_list)
ablation_df['R2_Drop'] = baseline_r2_mean - ablation_df['R2 (Mean)']
ablation_df['MAE_Increase'] = ablation_df['MAE (Mean)'] - baseline_mae_mean

baseline_row = ablation_df[ablation_df['Removed Feature'] == 'None (Full Model)']
other_rows = ablation_df[ablation_df['Removed Feature'] != 'None (Full Model)']
other_rows_sorted = other_rows.sort_values(by='R2_Drop', ascending=False)
final_ablation_summary = pd.concat([baseline_row, other_rows_sorted], ignore_index=True)

num_folds = kf_stack.get_n_splits()
r2_fold_cols = [f'R2_Fold_{i}' for i in range(1, num_folds + 1)]
mae_fold_cols = [f'MAE_Fold_{i}' for i in range(1, num_folds + 1)]
display_cols = (
        ['Removed Feature', 'R2 (Mean)', 'R2_Drop', 'MAE (Mean)', 'MAE_Increase', 'R2 (Std)', 'MAE (Std)'] +
        r2_fold_cols + mae_fold_cols
)
final_ablation_summary = final_ablation_summary[display_cols]

print("\n" + "=" * 80)
print("DETAILED ABLATION STUDY SUMMARY (Sorted by Importance)")
print("=" * 80 + "\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
    print(final_ablation_summary)

output_path = os.path.join(ensemble_models_path, 'detailed_ablation_study_summary_with_folds.csv')
final_ablation_summary.to_csv(output_path, index=False)
print(f"\nDetailed ablation study summary with per-fold results saved to: {output_path}")

# %% md
## PHASE 4: Final Model Optimization and Inverse Prediction
# %%
## PHASE 4: Final Model Optimization and Inverse Prediction
from scipy.optimize import differential_evolution

print("\n" + "=" * 80)
print("PHASE 4: Final Model Optimization and Inverse Prediction")
print("=" * 80 + "\n")

# --- Step 1: Create the OPTIMIZED Final Model based on Ablation Study ---
print("--- 1. Creating the Optimized Ensemble Model ---")
features_to_remove = final_ablation_summary[final_ablation_summary['R2_Drop'] < 0]['Removed Feature'].tolist()
original_feature_names = ensemble_feature_names

if features_to_remove:
    print(f"Based on the ablation study, the following features will be REMOVED to improve the model:")
    for feat in features_to_remove:
        print(f"  - {feat}")

    indices_to_remove = [original_feature_names.index(f) for f in features_to_remove]
    optimized_feature_names = [f for f in original_feature_names if f not in features_to_remove]
    X_ensemble_input_optimized = np.delete(X_ensemble_input, indices_to_remove, axis=1)
    optimized_meta_feature_names = [f for f in meta_feature_names if f not in features_to_remove]
else:
    print("Ablation study found no features that harm performance. Using the original full ensemble.")
    optimized_feature_names = original_feature_names
    X_ensemble_input_optimized = X_ensemble_input
    optimized_meta_feature_names = meta_feature_names

print(f"\nFinal model will be trained on {len(optimized_feature_names)} features: {optimized_feature_names}")
final_optimized_ensemble_model = clone(best_meta_model_prototype)

print("\nFitting the final OPTIMIZED meta-model on the refined feature set...")
final_optimized_ensemble_model.fit(X_ensemble_input_optimized, Y_ensemble_target)
print("Final Optimized Model is trained and ready.")

# =============================================================================
# ## MODIFICATION: The inverse prediction part is only applicable if there are
# ## environmental variables to optimize. We check if `other_features_names` is empty.
# =============================================================================
if other_features_names:
    # --- Step 2: User-defined inputs for Optimization ---
    target_smiles = 'C1=CC(=C(C(=C1)O)O)C(=O)O'
    target_concentration = 0.01  # mg/L, make sure units are consistent with training
    print("\n--- 2. Inverse Prediction Setup ---")
    print(f"Target Pollutant SMILES: {target_smiles}")
    print(f"Target Pollutant Concentration: {target_concentration} (ensure units match training data)")


    # --- Step 3: Define the Prediction Function (using the OPTIMIZED model) and Search Space ---
    def predict_kob_for_optimization(conditions, smiles, concentration):
        # MODIFICATION: Conditional unpacking of conditions
        if file_name != 'FeS':
            ph, temp, o3_conc = conditions  # Assuming the third condition is O3 concentration
        else:
            ph, temp = conditions
            o3_conc = None  # O3 is not an input condition for 'FeS' case

        mol_transformer = MolFromSmilesTransformer()  # Instantiate here or ensure it's thread-safe if global
        mol_list = mol_transformer.transform([smiles])
        if not mol_list or not mol_list[0]:
            # print(f"Warning: Invalid SMILES '{smiles}' during optimization.")
            return 1e10  # Return large positive value if SMILES is invalid (bad for maximization)
        mol = mol_list[0]

        meta_features_list = []
        for fp_name in optimized_meta_feature_names:
            fp_processor_info = final_base_models_and_processors[fp_name]
            fp_calc = fp_processor_info['fp_calculator']

            input_mol_for_fp_calc = mol  # Default to 2D mol

            if fp_processor_info['needs_conformers']:
                # Instantiate ConformerGenerator here for safety if not thread-safe,
                # or if it maintains state that could conflict.
                # For skfp, instantiating per call with num_conformers=1 is generally fine.
                conf_gen = ConformerGenerator(num_conformers=1, random_state=42)
                mol_3d_list = conf_gen.transform([mol])  # Pass as list
                if not mol_3d_list or not mol_3d_list[0]:
                    # print(f"Warning: Conformer generation failed for {smiles} with {fp_name}.")
                    return 1e10
                mol_3d = mol_3d_list[0]
                if mol_3d.GetNumConformers() == 0:
                    # print(f"Warning: No conformers generated for {smiles} with {fp_name}.")
                    return 1e10
                input_mol_for_fp_calc = mol_3d

            try:
                fp_raw = fp_calc.transform([input_mol_for_fp_calc])  # Pass as list
                fp_vector = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw).flatten()
            except Exception as e:
                # print(f"Warning: Fingerprint calculation failed for {smiles} with {fp_name}: {e}")
                return 1e10

            # MODIFICATION: Base model input is ONLY the fingerprint vector.
            # Environmental variables are not included here.
            full_feature_vector_for_base = fp_vector.reshape(1, -1)

            try:
                scaled_features_for_base = fp_processor_info['scaler'].transform(full_feature_vector_for_base)
                imputed_features_for_base = fp_processor_info['imputer'].transform(scaled_features_for_base)
                base_model_prediction = fp_processor_info['model'].predict(imputed_features_for_base)
                meta_features_list.append(base_model_prediction[0])
            except Exception as e:
                # print(f"Warning: Base model prediction failed for {fp_name}: {e}")
                return 1e10

        # These are the OOF-like predictions for the meta-features
        meta_features_for_ensemble = np.array(meta_features_list).reshape(1, -1)

        # Prepare environmental features for the final OPTIMIZED ensemble model
        # These are based on the current 'conditions' and fixed 'concentration'

        # First, construct the full set of environmental features in their original canonical order
        if file_name != 'FeS':
            # Original order: pH, Temp, Pollutant Con, O3
            all_current_env_features_original_order = np.array([ph, temp, concentration, o3_conc]).reshape(1, -1)
            original_env_order_list = ['pH', 'Temp', 'Pollutant Con', 'O3']
        else:
            # Original order: pH, Temp, Pollutant Con, S-Fe mol, S-nZVI Con
            # (S-Fe mol and S-nZVI Con are fixed at 0 for this optimization)
            all_current_env_features_original_order = np.array([ph, temp, concentration, 0, 0]).reshape(1, -1)
            original_env_order_list = ['pH', 'Temp', 'Pollutant Con', 'S-Fe mol', 'S-nZVI Con']

        # `env_feature_names_in_optimized_model` are those env features that survived ablation
        env_feature_names_in_optimized_model = [f for f in optimized_feature_names if
                                                f not in optimized_meta_feature_names]

        if env_feature_names_in_optimized_model:
            # Get indices of these surviving env features from their canonical original_env_order_list
            env_indices_to_select = [original_env_order_list.index(name) for name in
                                     env_feature_names_in_optimized_model]

            # Select these features from all_current_env_features_original_order
            selected_env_features_for_ensemble = all_current_env_features_original_order[:, env_indices_to_select]

            # Combine with meta-features
            ensemble_input_vector = np.concatenate((meta_features_for_ensemble, selected_env_features_for_ensemble),
                                                   axis=1)
        else:
            # No environmental features are used by the final optimized ensemble model
            ensemble_input_vector = meta_features_for_ensemble

        # Final prediction using the optimized ensemble model
        try:
            final_kob_prediction = final_optimized_ensemble_model.predict(ensemble_input_vector)[0]
        except Exception as e:
            # print(f"Warning: Final ensemble prediction failed: {e}")
            return 1e10

        return -final_kob_prediction  # differential_evolution minimizes; we want to maximize Kob


    # Define bounds for optimization
    # Ensure these align with the 'conditions' expected by predict_kob_for_optimization
    if file_name != 'FeS':
        # Expecting pH, Temp, O3 concentration
        bounds = [
            (data['pH'].min(), data['pH'].max()),  # pH
            (data['Temp'].min(), data['Temp'].max()),  # Temperature
            (data['O3'].min(), data['O3'].max())  # O3 concentration
        ]
        variable_names_for_print = ["pH", "Temp", "O3 (mg/L)"]
    else:
        # Expecting pH, Temp (O3 not applicable for FeS)
        bounds = [
            (data['pH'].min(), data['pH'].max()),  # pH
            (data['Temp'].min(), data['Temp'].max())  # Temperature
        ]
        variable_names_for_print = ["pH", "Temp"]

    print("\n--- 3. Running Optimization Search ---")
    print(f"Search bounds: {bounds}")

    # --- Step 4: Run the Optimization ---
    result = differential_evolution(
        func=predict_kob_for_optimization,
        bounds=bounds,
        args=(target_smiles, target_concentration),  # These are fixed for the optimization
        strategy='best1bin', maxiter=200, popsize=20, tol=0.01,
        mutation=(0.5, 1), recombination=0.7, disp=True, seed=42
    )

    # --- Step 5: Display the Results ---
    if result.success:
        optimal_conditions = result.x
        max_kob = -result.fun  # Since we minimized -Kob
        print("\n" + "-" * 50)
        print("Optimization Finished Successfully!")
        for i, name in enumerate(variable_names_for_print):
            print(f"Optimal {name}: {optimal_conditions[i]:.2f}")
        print("-" * 50)
        print(f"Predicted Maximum Kob: {max_kob:.4f} [min-1]")
        print("-" * 50)
    else:
        print("\nOptimization did not converge or failed.")
        print(f"Message: {result.message}")
else:
    print("\n--- 2. SKIPPING Inverse Prediction ---")
    print("This step is not applicable because the dataset does not contain environmental variables to optimize.")
    # Define bounds as empty for artifact saving
    bounds = []
# =============================================================================
# ## END OF MODIFICATION
# =============================================================================
# %% md
## PHASE 5: SHAP Analysis and Visualization
# %%
## PHASE 5: SHAP Analysis and Visualization
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Import fingerprint classes for type checking
try:
    from skfp.fingerprints import ECFP, MorganFingerprint  # Add others like CircularFingerprint if used

    SKFP_FP_CLASSES_IMPORTED = True
except ImportError:
    print(
        "Warning: skfp fingerprint classes (ECFP, MorganFingerprint) could not be imported. Type checking for ECFP/Morgan models will rely on name.")
    ECFP, MorganFingerprint = type(None), type(None)  # Define as NoneType for isinstance checks to fail gracefully
    SKFP_FP_CLASSES_IMPORTED = False

# RDKit check
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit is not installed. ECFP visualization will be skipped.")

print("\n" + "=" * 80)
print("PHASE 5: SHAP Analysis for Model Interpretability and Visualization")
print("=" * 80 + "\n")

# Variables to store results from PART 2, to be potentially used by PART 3
part2_analyzed_model_name = None
part2_analyzed_model_is_ecfp_type = False
part2_shap_values = None
# part2_X_input_for_shap = None # Data is within shap_values.data
part2_all_feature_names = None
part2_fp_matrix = None
part2_input_mols_for_fp_calc = None
part2_fp_calculator_instance = None
part2_bar_data_df = None  # Feature importance data from PART 2
part2_model_object = None  # Model instance analyzed in PART 2
part2_shap_output_path = None  # Base path for SHAP outputs of the model analyzed in part 2

# =============================================================================
# PART 1: 分析性能最佳的最终集成模型
# =============================================================================
print("\n--- 1. Analyzing the Best Performing Ensemble Model ---")
most_influential_base_model_name_from_ensemble = None  # Determined in this part

if 'ensemble_summary_df' not in locals() or \
        'tuned_meta_models' not in locals() or \
        'X_ensemble_input' not in locals() or \
        'ensemble_feature_names' not in locals() or \
        'meta_feature_names' not in locals() or \
        'ensemble_models_path' not in locals():
    print("Error: Prerequisite variables from earlier phases are missing for Ensemble SHAP.")
    can_proceed_to_part2_selection = False
else:
    can_proceed_to_part2_selection = True
    tuned_models_summary = ensemble_summary_df[ensemble_summary_df['Ensemble Type'].str.startswith('Tuned_')].copy()

    if tuned_models_summary.empty:
        print("No tuned ensemble models found. Skipping ensemble SHAP analysis.")
        can_proceed_to_part2_selection = False
    else:
        # MODIFICATION: Use the correct column name 'R2 Test (Mean)' to find the best model
        best_model_info = tuned_models_summary.loc[tuned_models_summary['R2 Test (Mean)'].idxmax()]
        best_ensemble_name_full = best_model_info['Ensemble Type']
        model_key = best_ensemble_name_full.replace('Tuned_', '')
        print(f"The best performing ensemble model is: '{best_ensemble_name_full}'")
        final_ensemble_model = tuned_meta_models.get(model_key)

        if final_ensemble_model:
            print("Creating SHAP explainer for the ensemble model...")
            # Adapt explainer based on model type (as per previous fixes)
            try:
                if hasattr(final_ensemble_model, 'coef_'):  # Linear models (e.g., RidgeCV)
                    explainer_ensemble = shap.LinearExplainer(final_ensemble_model, X_ensemble_input,
                                                              feature_names=ensemble_feature_names)
                elif hasattr(final_ensemble_model, 'feature_importances_'):  # Tree-based models
                    explainer_ensemble = shap.TreeExplainer(final_ensemble_model, X_ensemble_input,
                                                            feature_names=ensemble_feature_names)  # Pass data for background
                else:  # Fallback for other model types (e.g., KernelExplainer or if model.predict is simple)
                    print("Using generic shap.Explainer for ensemble model (might be slower).")
                    explainer_ensemble = shap.Explainer(final_ensemble_model.predict, X_ensemble_input,
                                                        feature_names=ensemble_feature_names)

                shap_values_ensemble = explainer_ensemble(X_ensemble_input)
                print("SHAP values calculated for ensemble model.")
            except Exception as e:
                print(f"Error creating SHAP explainer or values for ensemble model: {e}")
                shap_values_ensemble = None
                can_proceed_to_part2_selection = False

            if shap_values_ensemble is not None:
                shap_ensemble_path = os.path.join(ensemble_models_path, 'shap_analysis_ensemble')
                os.makedirs(shap_ensemble_path, exist_ok=True)
                print(f"Saving ensemble SHAP plots to: {shap_ensemble_path}")

                print("Exporting raw SHAP data for the ensemble model...")
                mean_abs_shap_ensemble = np.abs(shap_values_ensemble.values).mean(axis=0)
                bar_data_df_ensemble = pd.DataFrame({
                    'feature': ensemble_feature_names,
                    'mean_abs_shap_value': mean_abs_shap_ensemble
                }).sort_values(by='mean_abs_shap_value', ascending=False)
                bar_data_path = os.path.join(shap_ensemble_path, 'shap_bar_plot_raw_data_ensemble.csv')
                bar_data_df_ensemble.to_csv(bar_data_path, index=False)
                print(f"  -> Saved bar plot data to: {bar_data_path}")

                # Beeswarm data export
                try:
                    beeswarm_data_df = pd.DataFrame(shap_values_ensemble.values, columns=ensemble_feature_names)
                    beeswarm_data_df['sample_id'] = beeswarm_data_df.index
                    beeswarm_data_df = beeswarm_data_df.melt(id_vars='sample_id', var_name='feature',
                                                             value_name='shap_value')

                    # Ensure shap_values_ensemble.data is accessible and correct
                    if shap_values_ensemble.data is not None and shap_values_ensemble.data.shape == X_ensemble_input.shape:
                        feature_values_for_beeswarm = pd.DataFrame(shap_values_ensemble.data,
                                                                   columns=ensemble_feature_names)
                    else:  # Fallback to X_ensemble_input if .data is not as expected
                        feature_values_for_beeswarm = pd.DataFrame(X_ensemble_input, columns=ensemble_feature_names)

                    feature_values_for_beeswarm['sample_id'] = feature_values_for_beeswarm.index
                    feature_values_for_beeswarm = feature_values_for_beeswarm.melt(id_vars='sample_id',
                                                                                   var_name='feature',
                                                                                   value_name='feature_value')

                    beeswarm_data_df = pd.merge(beeswarm_data_df, feature_values_for_beeswarm,
                                                on=['sample_id', 'feature'])
                    beeswarm_data_path = os.path.join(shap_ensemble_path, 'shap_beeswarm_plot_raw_data_ensemble.csv')
                    beeswarm_data_df.to_csv(beeswarm_data_path, index=False)
                    print(f"  -> Saved beeswarm plot data to: {beeswarm_data_path}")
                except Exception as e_beeswarm:
                    print(f"  -> Error exporting beeswarm data for ensemble: {e_beeswarm}")

                plt.figure()
                shap.summary_plot(shap_values_ensemble, plot_type="bar", show=False)
                plt.title(f'Feature Importance for {best_ensemble_name_full}')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_ensemble_path, f'shap_bar_ensemble.png'))
                plt.close()

                plt.figure()
                shap.summary_plot(shap_values_ensemble, show=False)
                plt.title(f'Feature Impact for {best_ensemble_name_full}')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_ensemble_path, f'shap_beeswarm_ensemble.png'))
                plt.close()
                print("Ensemble SHAP analysis complete.")

                meta_features_shap_df = bar_data_df_ensemble[bar_data_df_ensemble['feature'].isin(meta_feature_names)]
                if not meta_features_shap_df.empty:
                    most_influential_base_model_name_from_ensemble = meta_features_shap_df.iloc[0]['feature']
                else:
                    print("Warning: Could not determine the most influential base model from ensemble SHAP values.")
                    # can_proceed_to_part2_selection remains true, but most_influential_base_model_name_from_ensemble is None
            else:  # shap_values_ensemble is None
                can_proceed_to_part2_selection = False  # Cannot determine influential model

        else:  # final_ensemble_model not found
            print(f"Could not find the model for key '{model_key}' in the tuned models dictionary.")
            can_proceed_to_part2_selection = False

# =============================================================================
# PART 2: 分析目标基础模型 (优先 ECFP/Morgan, 后备为集成中贡献最大的模型)
# =============================================================================
print("\n--- 2. Analyzing Target Base Model (Prioritizing ECFP/Morgan) ---")

# --- Determine the target model for PART 2 ---
_model_to_analyze_in_part2_name = None
_model_to_analyze_in_part2_details = None
_model_to_analyze_in_part2_is_ecfp = False

if 'final_base_models_and_processors' in locals() and \
        'K' in locals() and 'other_features' in locals() and \
        'mols_actually_with_conformers' in locals() and \
        'indices_with_conformers_relative_to_mols2d' in locals() and \
        'SMILES' in locals() and 'mols_2d' in locals() and \
        'other_features_names' in locals() and 'base_models_path' in locals():

    # 1. Try to find an ECFP/Morgan model
    for model_name_iter, model_info_iter in final_base_models_and_processors.items():
        fp_calc_iter = model_info_iter.get('fp_calculator')
        is_ecfp_type_check = False
        if fp_calc_iter:
            if SKFP_FP_CLASSES_IMPORTED and isinstance(fp_calc_iter, (ECFP, MorganFingerprint)):
                is_ecfp_type_check = True
            else:  # Fallback to name check
                fp_calc_type_name = type(fp_calc_iter).__name__.lower()
                if 'ecfp' in fp_calc_type_name or 'morgan' in fp_calc_type_name or 'circularfingerprint' in fp_calc_type_name:
                    is_ecfp_type_check = True

        if is_ecfp_type_check:
            _model_to_analyze_in_part2_name = model_name_iter
            _model_to_analyze_in_part2_details = model_info_iter
            _model_to_analyze_in_part2_is_ecfp = True
            print(f"Prioritizing ECFP/Morgan model for PART 2 analysis: '{_model_to_analyze_in_part2_name}'")
            break

    # 2. If no ECFP/Morgan model found, fall back to the most influential from ensemble (if available and PART 1 ran)
    if not _model_to_analyze_in_part2_name and can_proceed_to_part2_selection and most_influential_base_model_name_from_ensemble:
        if most_influential_base_model_name_from_ensemble in final_base_models_and_processors:
            print(
                f"No ECFP/Morgan model found. Using most influential model from ensemble for PART 2: '{most_influential_base_model_name_from_ensemble}'")
            _model_to_analyze_in_part2_name = most_influential_base_model_name_from_ensemble
            _model_to_analyze_in_part2_details = final_base_models_and_processors[
                most_influential_base_model_name_from_ensemble]
            # Check if this fallback is ECFP (it shouldn't be if the first loop was exhaustive)
            fp_calc_fallback = _model_to_analyze_in_part2_details.get('fp_calculator')
            is_ecfp_fallback_check = False
            if fp_calc_fallback:
                if SKFP_FP_CLASSES_IMPORTED and isinstance(fp_calc_fallback, (ECFP, MorganFingerprint)):
                    is_ecfp_fallback_check = True
                else:
                    fp_calc_type_name = type(fp_calc_fallback).__name__.lower()
                    if 'ecfp' in fp_calc_type_name or 'morgan' in fp_calc_type_name or 'circularfingerprint' in fp_calc_type_name:
                        is_ecfp_fallback_check = True
            _model_to_analyze_in_part2_is_ecfp = is_ecfp_fallback_check  # Set whether this fallback is ECFP
        else:
            print(
                f"Most influential model '{most_influential_base_model_name_from_ensemble}' not found in final_base_models_and_processors.")
            _model_to_analyze_in_part2_name = None  # Ensure it's None

    elif not _model_to_analyze_in_part2_name:  # If ECFP not found and influential also not usable
        print(
            "No ECFP/Morgan model found, and no suitable most influential model from ensemble. Skipping PART 2 SHAP analysis.")
        _model_to_analyze_in_part2_name = None

    # --- Proceed with SHAP analysis if a target model for PART 2 was determined ---
    if _model_to_analyze_in_part2_name and _model_to_analyze_in_part2_details:
        part2_analyzed_model_name = _model_to_analyze_in_part2_name
        part2_analyzed_model_is_ecfp_type = _model_to_analyze_in_part2_is_ecfp

        print(
            f"Analyzing in PART 2: '{part2_analyzed_model_name}' (Is ECFP/Morgan: {part2_analyzed_model_is_ecfp_type})")

        part2_model_object = _model_to_analyze_in_part2_details['model']
        part2_fp_calculator_instance = _model_to_analyze_in_part2_details['fp_calculator']
        scaler = _model_to_analyze_in_part2_details['scaler']
        imputer = _model_to_analyze_in_part2_details['imputer']
        needs_conformers = _model_to_analyze_in_part2_details['needs_conformers']

        # Reconstruct input data for this specific base model
        _Y_for_model = K
        # MODIFICATION: We don't need to subset environmental features, they are not used.
        # _current_other_features = other_features

        if needs_conformers:
            part2_input_mols_for_fp_calc = mols_actually_with_conformers
            _indices_for_fp = indices_with_conformers_relative_to_mols2d
            _Y_for_model = K[_indices_for_fp]
        elif hasattr(part2_fp_calculator_instance,
                     '__class__') and "Lingo" in part2_fp_calculator_instance.__class__.__name__:
            part2_input_mols_for_fp_calc = SMILES  # Assuming Lingo takes SMILES
        else:  # Default to 2D mols for ECFP or other non-conformer FPs
            part2_input_mols_for_fp_calc = mols_2d

        _FP_raw = part2_fp_calculator_instance.transform(part2_input_mols_for_fp_calc)
        part2_fp_matrix = np.asarray(_FP_raw.toarray() if hasattr(_FP_raw, "toarray") else _FP_raw)

        # MODIFICATION: The input for the base model is only the fingerprint matrix.
        _feature_w_fp_unscaled = part2_fp_matrix
        _X_scaled = scaler.transform(_feature_w_fp_unscaled)
        _X_imputed_for_shap = imputer.transform(_X_scaled)  # This is part2_X_input_for_shap

        # MODIFICATION: The feature names for the base model do not include environmental features.
        _fp_feature_names = [f'{part2_analyzed_model_name}_{i}' for i in range(part2_fp_matrix.shape[1])]
        part2_all_feature_names = _fp_feature_names

        # Run SHAP analysis for the targeted base model
        print(f"Running SHAP for base model: {part2_analyzed_model_name}")
        try:
            if hasattr(part2_model_object, 'feature_importances_'):  # Tree-based
                explainer_base = shap.TreeExplainer(part2_model_object, _X_imputed_for_shap)  # Provide background data
            elif hasattr(part2_model_object, 'coef_'):  # Linear
                explainer_base = shap.LinearExplainer(part2_model_object, _X_imputed_for_shap)
            else:
                print(
                    f"Warning: Model type for {part2_analyzed_model_name} not recognized as tree/linear. Using generic shap.Explainer.")
                explainer_base = shap.Explainer(part2_model_object.predict, _X_imputed_for_shap)

            part2_shap_values = explainer_base(_X_imputed_for_shap)  # Pass data to get SHAP values for
            # Ensure feature names are in the explanation object
            if part2_shap_values.feature_names is None and part2_all_feature_names is not None:
                part2_shap_values.feature_names = part2_all_feature_names
            print(f"SHAP values calculated for {part2_analyzed_model_name}.")

        except Exception as e:
            print(f"Error during SHAP analysis for {part2_analyzed_model_name}: {e}")
            part2_shap_values = None  # Ensure it's None on error

        if part2_shap_values:
            part2_shap_output_path = os.path.join(base_models_path, part2_analyzed_model_name,
                                                  'shap_analysis_part2_target')
            os.makedirs(part2_shap_output_path, exist_ok=True)
            print(f"Saving SHAP plots for '{part2_analyzed_model_name}' to: {part2_shap_output_path}")

            print(f"Exporting raw SHAP data for '{part2_analyzed_model_name}'...")
            _mean_abs_shap_base = np.abs(part2_shap_values.values).mean(axis=0)
            part2_bar_data_df = pd.DataFrame({
                'feature': part2_all_feature_names,
                'mean_abs_shap_value': _mean_abs_shap_base
            }).sort_values(by='mean_abs_shap_value', ascending=False)
            _bar_data_path_base = os.path.join(part2_shap_output_path, 'shap_bar_plot_raw_data.csv')
            part2_bar_data_df.to_csv(_bar_data_path_base, index=False)
            print(f"  -> Saved bar plot data to: {_bar_data_path_base}")

            # Beeswarm data export
            try:
                _beeswarm_data_df_base = pd.DataFrame(part2_shap_values.values, columns=part2_all_feature_names)
                _beeswarm_data_df_base['sample_id'] = _beeswarm_data_df_base.index
                _beeswarm_data_df_base = _beeswarm_data_df_base.melt(id_vars='sample_id', var_name='feature',
                                                                     value_name='shap_value')

                _feature_values_for_beeswarm_base = pd.DataFrame(part2_shap_values.data,
                                                                 columns=part2_all_feature_names)  # Use .data from SHAP obj
                _feature_values_for_beeswarm_base['sample_id'] = _feature_values_for_beeswarm_base.index
                _feature_values_for_beeswarm_base = _feature_values_for_beeswarm_base.melt(id_vars='sample_id',
                                                                                           var_name='feature',
                                                                                           value_name='feature_value')

                _beeswarm_data_df_base = pd.merge(_beeswarm_data_df_base, _feature_values_for_beeswarm_base,
                                                  on=['sample_id', 'feature'])
                _beeswarm_data_path_base = os.path.join(part2_shap_output_path, 'shap_beeswarm_plot_raw_data.csv')
                _beeswarm_data_df_base.to_csv(_beeswarm_data_path_base, index=False)
                print(f"  -> Saved beeswarm plot data to: {_beeswarm_data_path_base}")
            except Exception as e_beeswarm_base:
                print(f"  -> Error exporting beeswarm data for {part2_analyzed_model_name}: {e_beeswarm_base}")

            plt.figure(figsize=(10, 8))
            shap.summary_plot(part2_shap_values, plot_type="bar", max_display=20, show=False)
            plt.title(f'Top 20 Feature Importance for {part2_analyzed_model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(part2_shap_output_path, 'shap_bar_plot.png'))
            plt.close()

            plt.figure(figsize=(10, 8))
            shap.summary_plot(part2_shap_values, max_display=20, show=False)
            plt.title(f'Top 20 Feature Impact for {part2_analyzed_model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(part2_shap_output_path, 'shap_beeswarm_plot.png'))
            plt.close()
            print(f"SHAP analysis for '{part2_analyzed_model_name}' complete.")
    else:  # No model determined for PART 2 or details missing
        print("Skipping PART 2 SHAP analysis as no suitable base model was identified or details are missing.")
else:  # Prerequisites for PART 2 missing
    print("Error: Prerequisite variables for PART 2 (Base Model SHAP) are missing. Skipping.")

# =============================================================================
# PART 3: 可视化重要的分子亚结构/特征 (ECFP/Morgan Specific)
# =============================================================================
print("\n--- 3. Visualizing Important Molecular Features (ECFP/Morgan Specific) ---")

if not RDKIT_AVAILABLE:
    print("RDKit is not available. Skipping ECFP visualization.")
elif 'final_base_models_and_processors' not in locals() or \
        'K' not in locals() or 'other_features' not in locals() or \
        'mols_actually_with_conformers' in locals() or \
        'indices_with_conformers_relative_to_mols2d' in locals() or \
        'SMILES' not in locals() or 'mols_2d' not in locals() or \
        'other_features_names' in locals() or 'base_models_path' not in locals():
    print("Error: Prerequisite variables for ECFP/Morgan SHAP visualization are missing. Skipping.")
else:
    _ecfp_model_name_for_viz = None
    _shap_values_for_viz = None
    _fp_matrix_for_viz = None
    _input_mols_for_viz_fp_calc = None
    _fp_calculator_for_viz = None
    _bar_data_df_for_viz = None  # This will hold feature importances for the viz target
    _shap_path_for_viz_plots = None  # Path for saving visualization plots

    # Check if PART 2 already analyzed an ECFP model and has valid results
    if part2_analyzed_model_name and part2_analyzed_model_is_ecfp_type and \
            part2_shap_values is not None and part2_bar_data_df is not None and \
            part2_fp_matrix is not None and part2_input_mols_for_fp_calc is not None and \
            part2_fp_calculator_instance is not None:

        print(f"Using SHAP results from PART 2 for ECFP model '{part2_analyzed_model_name}' for visualization.")
        _ecfp_model_name_for_viz = part2_analyzed_model_name
        _shap_values_for_viz = part2_shap_values  # Not directly used for bit viz, but bar_data is
        _fp_matrix_for_viz = part2_fp_matrix
        _input_mols_for_viz_fp_calc = part2_input_mols_for_fp_calc
        _fp_calculator_for_viz = part2_fp_calculator_instance
        _bar_data_df_for_viz = part2_bar_data_df
        # Use a subfolder of part2_shap_output_path or a new one
        _shap_path_for_viz_plots = os.path.join(base_models_path, _ecfp_model_name_for_viz,
                                                'ecfp_shap_visualizations_from_part2')
    else:
        print(
            "PART 2 did not analyze an ECFP model or results are incomplete. PART 3 will search and (re)analyze one if found.")
        # --- Fallback: Find an ECFP/Morgan model and run SHAP for it in PART 3 ---
        _part3_target_model_name = None
        _part3_target_model_details = None

        for model_name_iter, model_info_iter in final_base_models_and_processors.items():
            fp_calc_iter = model_info_iter.get('fp_calculator')
            is_ecfp_type_check = False
            if fp_calc_iter:
                if SKFP_FP_CLASSES_IMPORTED and isinstance(fp_calc_iter, (ECFP, MorganFingerprint)):
                    is_ecfp_type_check = True
                else:
                    fp_calc_type_name = type(fp_calc_iter).__name__.lower()
                    if 'ecfp' in fp_calc_type_name or 'morgan' in fp_calc_type_name or 'circularfingerprint' in fp_calc_type_name:
                        is_ecfp_type_check = True
            if is_ecfp_type_check:
                _part3_target_model_name = model_name_iter
                _part3_target_model_details = model_info_iter
                print(f"PART 3 found ECFP/Morgan model for dedicated visualization: '{_part3_target_model_name}'")
                break

        if _part3_target_model_name and _part3_target_model_details:
            _ecfp_model_name_for_viz = _part3_target_model_name
            _model_obj = _part3_target_model_details['model']
            _fp_calculator_for_viz = _part3_target_model_details['fp_calculator']
            _scaler = _part3_target_model_details['scaler']
            _imputer = _part3_target_model_details['imputer']
            _needs_conformers = _part3_target_model_details['needs_conformers']  # Should be False for ECFP

            # Reconstruct input for this ECFP model for SHAP
            _input_mols_for_viz_fp_calc = mols_2d  # ECFP typically uses 2D
            if _needs_conformers:  # Should not happen for ECFP but handle
                _input_mols_for_viz_fp_calc = mols_actually_with_conformers

            print(f"Calculating fingerprints using {type(_fp_calculator_for_viz).__name__} for PART 3 viz SHAP...")
            _FP_raw_viz = _fp_calculator_for_viz.transform(_input_mols_for_viz_fp_calc)
            _fp_matrix_for_viz = np.asarray(_FP_raw_viz.toarray() if hasattr(_FP_raw_viz, "toarray") else _FP_raw_viz)

            # MODIFICATION: Input for base model is only fingerprints.
            _feat_w_fp_unscaled_viz = _fp_matrix_for_viz
            _X_scaled_viz = _scaler.transform(_feat_w_fp_unscaled_viz)
            _X_imputed_for_shap_viz = _imputer.transform(_X_scaled_viz)

            # MODIFICATION: Feature names for base model are only fingerprint names.
            _fp_feat_names_viz = [f'{_ecfp_model_name_for_viz}_{i}' for i in range(_fp_matrix_for_viz.shape[1])]
            _all_feat_names_viz = _fp_feat_names_viz

            print(f"Running SHAP explainer for ECFP model '{_ecfp_model_name_for_viz}' in PART 3...")
            _temp_shap_values_viz = None
            try:
                if hasattr(_model_obj, 'feature_importances_'):
                    explainer_viz = shap.TreeExplainer(_model_obj, _X_imputed_for_shap_viz)
                elif hasattr(_model_obj, 'coef_'):
                    explainer_viz = shap.LinearExplainer(_model_obj, _X_imputed_for_shap_viz)
                else:
                    explainer_viz = shap.Explainer(_model_obj.predict, _X_imputed_for_shap_viz)
                _temp_shap_values_viz = explainer_viz(_X_imputed_for_shap_viz)
                if _temp_shap_values_viz.feature_names is None: _temp_shap_values_viz.feature_names = _all_feat_names_viz
            except Exception as e:
                print(f"Error during SHAP for '{_ecfp_model_name_for_viz}' in PART 3: {e}")

            if _temp_shap_values_viz:
                _mean_abs_shap_viz = np.abs(_temp_shap_values_viz.values).mean(axis=0)
                _bar_data_df_for_viz = pd.DataFrame({
                    'feature': _all_feat_names_viz,
                    'mean_abs_shap_value': _mean_abs_shap_viz
                }).sort_values(by='mean_abs_shap_value', ascending=False)
                _shap_path_for_viz_plots = os.path.join(base_models_path, _ecfp_model_name_for_viz,
                                                        'ecfp_shap_visualizations_from_part3_analysis')
            else:
                _ecfp_model_name_for_viz = None  # Failed to get SHAP
        else:  # No ECFP model found by PART 3 either
            print("No ECFP/Morgan model found by PART 3. Skipping ECFP visualization.")
            _ecfp_model_name_for_viz = None

    # --- Actual Visualization step ---
    if _ecfp_model_name_for_viz and _bar_data_df_for_viz is not None and \
            _fp_matrix_for_viz is not None and _input_mols_for_viz_fp_calc is not None and \
            _fp_calculator_for_viz is not None and _shap_path_for_viz_plots is not None:

        os.makedirs(_shap_path_for_viz_plots, exist_ok=True)
        print(f"Visualizing top ECFP features for '{_ecfp_model_name_for_viz}' using data prepared.")
        print(f"Plots will be saved to: {_shap_path_for_viz_plots}")

        # MODIFICATION: Now that base models have no env vars, we don't need to filter them out. All features are FP features.
        fp_shap_df_viz = _bar_data_df_for_viz
        top_n_features_viz = 10
        top_fp_features_viz = fp_shap_df_viz.head(top_n_features_viz)

        if top_fp_features_viz.empty:
            print(f"No fingerprint features found for '{_ecfp_model_name_for_viz}' in SHAP results. Cannot visualize.")
        else:
            # Get Morgan/ECFP parameters (handle skfp's new `fp_size` for nBits)
            radius_viz = getattr(_fp_calculator_for_viz, 'radius', 2)  # Default ECFP4 (radius 2)
            nBits_viz = getattr(_fp_calculator_for_viz, 'fp_size', getattr(_fp_calculator_for_viz, 'n_bits', 2048))
            use_chirality_viz = getattr(_fp_calculator_for_viz, 'use_chirality', False)
            use_features_viz = getattr(_fp_calculator_for_viz, 'use_features',
                                       False)  # ECFP default False (atom types), Morgan default False (atom types)

            print(
                f"Using ECFP/Morgan parameters for viz: radius={radius_viz}, nBits/fp_size={nBits_viz}, chirality={use_chirality_viz}, useFeatures={use_features_viz}")

            for _, row in top_fp_features_viz.iterrows():
                feature_name = row['feature']
                try:
                    bit_id = int(feature_name.split('_')[-1])
                except ValueError:
                    print(f"  -> Warning: Could not parse bit_id from feature_name '{feature_name}'. Skipping.")
                    continue
                bit_importance = row['mean_abs_shap_value']

                molecule_indices_with_bit = np.where(_fp_matrix_for_viz[:, bit_id] == 1)[0]

                if len(molecule_indices_with_bit) > 0:
                    example_mol_idx_in_fp_input = molecule_indices_with_bit[0]
                    mol_input_for_draw = _input_mols_for_viz_fp_calc[example_mol_idx_in_fp_input]

                    example_rdkit_mol = None
                    if isinstance(mol_input_for_draw, str):  # SMILES
                        example_rdkit_mol = Chem.MolFromSmiles(mol_input_for_draw)
                    elif isinstance(mol_input_for_draw, Chem.Mol):  # RDKit Mol object
                        example_rdkit_mol = mol_input_for_draw

                    if not example_rdkit_mol:
                        print(
                            f"  -> Warning: Could not obtain valid RDKit Mol for bit {bit_id} from input type {type(mol_input_for_draw)}. Skipping.")
                        continue

                    bit_info = {}
                    _ = AllChem.GetMorganFingerprintAsBitVect(example_rdkit_mol,
                                                              radius=radius_viz,
                                                              nBits=nBits_viz,
                                                              useChirality=use_chirality_viz,
                                                              useFeatures=use_features_viz,
                                                              bitInfo=bit_info)
                    if bit_id in bit_info:
                        try:
                            img = Draw.DrawMorganBit(example_rdkit_mol, bit_id, bit_info, useSVG=False)
                            importance_str = f"{bit_importance:.3f}".replace('.', '_')
                            output_png_path = os.path.join(_shap_path_for_viz_plots,
                                                           f'ecfp_bit_{bit_id}_shap_{importance_str}.png')
                            if hasattr(img, 'save'):  # PIL Image
                                img.save(output_png_path)
                                print(f"  -> Saved ECFP 2D visualization: {output_png_path}")
                            else:  # Should not happen with useSVG=False
                                print(f"  -> Warning: DrawMorganBit did not return a saveable image for bit {bit_id}.")
                        except Exception as e_draw:
                            print(
                                f"  -> Error during drawing/saving for bit {bit_id} ({Chem.MolToSmiles(example_rdkit_mol)}): {e_draw}. Skipping.")
                    else:
                        print(
                            f"  -> Warning: Bit {bit_id} not found in bit_info for example molecule {Chem.MolToSmiles(example_rdkit_mol) if example_rdkit_mol else 'N/A'} (radius {radius_viz}, nBits {nBits_viz}, useFeatures {use_features_viz}). Skipping.")
                else:
                    print(
                        f"  -> Warning: No molecule found with feature bit {bit_id} in the dataset for '{_ecfp_model_name_for_viz}'. Skipping.")
    elif not _ecfp_model_name_for_viz:  # If no ECFP model could be processed for viz
        print("Visualization skipped: No suitable ECFP/Morgan model and its SHAP data available for visualization.")
    # Implicitly, if RDKit wasn't available or prerequisites were missing, earlier messages cover it.

print("\nAll SHAP analyses and visualizations (attempted) finished.")

# =============================================================================
# 【新增代码】PART 4: 可视化重要的分子亚结构 (Topological Torsion Specific)
# =============================================================================
print("\n--- 4. Visualizing Important Molecular Features (Topological Torsion Specific) ---")

# 此部分在PART 2中分析的模型是'TopologicalTorsion'时运行
if part2_analyzed_model_name == 'TopologicalTorsion' and RDKIT_AVAILABLE and \
        part2_shap_values is not None and part2_bar_data_df is not None and \
        part2_fp_matrix is not None and part2_input_mols_for_fp_calc is not None and \
        part2_fp_calculator_instance is not None and part2_shap_output_path is not None:

    print(f"Visualizing top Topological Torsion features for '{part2_analyzed_model_name}'.")

    # 创建专门用于存放扭转可视化的子目录
    torsion_viz_path = os.path.join(part2_shap_output_path, 'torsion_visualizations')
    os.makedirs(torsion_viz_path, exist_ok=True)
    print(f"Plots will be saved to: {torsion_viz_path}")

    # 从SHAP分析结果中获取最重要的特征（bits）
    top_n_features = 10
    top_fp_features = part2_bar_data_df.head(top_n_features)

    if top_fp_features.empty:
        print(f"No fingerprint features found for '{part2_analyzed_model_name}' in SHAP results. Cannot visualize.")
    else:
        # 从指纹计算器实例中获取参数
        nBits = getattr(part2_fp_calculator_instance, 'fp_size', 2048)
        torsion_size = getattr(part2_fp_calculator_instance, 'target_size', 4)
        print(f"Using TopologicalTorsion parameters for viz: nBits/fp_size={nBits}, target_size={torsion_size}")

        for index, row in top_fp_features.iterrows():
            feature_name = row['feature']
            bit_importance = row['mean_abs_shap_value']

            try:
                bit_id = int(feature_name.split('_')[-1])
            except (ValueError, IndexError):
                print(f"  -> 警告: 无法从特征名称 '{feature_name}' 解析bit_id。跳过。")
                continue

            # 查找数据集中包含此bit的一个分子作为示例
            molecule_indices_with_bit = np.where(part2_fp_matrix[:, bit_id] == 1)[0]

            if len(molecule_indices_with_bit) > 0:
                example_mol_idx = molecule_indices_with_bit[0]
                example_rdkit_mol = part2_input_mols_for_fp_calc[example_mol_idx]

                if not isinstance(example_rdkit_mol, Chem.Mol):
                    print(f"  -> 警告: 无法为bit {bit_id}获取有效的RDKit Mol对象。跳过。")
                    continue

                # 为这个示例分子重新计算指纹以获取bitInfo
                bit_info = {}
                _ = AllChem.GetTopologicalTorsionFingerprintAsBitVect(
                    example_rdkit_mol,
                    nBits=nBits,
                    targetSize=torsion_size,
                    bitInfo=bit_info
                )

                if bit_id in bit_info:
                    # bit_info[bit_id] 是一个列表，其中每个元素都是一个扭转路径（4个原子索引的元组）
                    # 我们只可视化第一个作为代表
                    torsion_atom_indices = bit_info[bit_id][0]

                    try:
                        legend_text = f'Bit: {bit_id} | Atoms: {torsion_atom_indices}\nSHAP: {bit_importance:.3f}'
                        img = draw_topological_torsion_substructure(
                            example_rdkit_mol,
                            list(torsion_atom_indices),
                            legend=legend_text,
                            use_svg=False  # 保存为PNG
                        )

                        if img:
                            importance_str = f"{bit_importance:.3f}".replace('.', '_')
                            rank = index + 1
                            output_png_path = os.path.join(torsion_viz_path,
                                                           f'rank_{rank}_torsion_bit_{bit_id}_shap_{importance_str}.png')
                            img.save(output_png_path)
                            print(f"  -> 已保存拓扑扭转2D可视化: {output_png_path}")
                        else:
                            print(f"  -> 警告: 无法为bit {bit_id}生成图像。")

                    except Exception as e_draw:
                        print(
                            f"  -> 错误: 绘制或保存bit {bit_id}时出错 ({Chem.MolToSmiles(example_rdkit_mol)}): {e_draw}。")
                else:
                    print(f"  -> 警告: 在示例分子的bit_info中未找到bit {bit_id}。这可能是由于哈希碰撞。")
            else:
                print(f"  -> 警告: 数据集中未找到包含特征bit {bit_id}的分子。")

else:
    print("跳过拓扑扭转可视化：模型不是'TopologicalTorsion'或缺少先决条件（RDKit, SHAP结果等）。")
# %% md
## PHASE 6: Save All Necessary Artifacts for Web App
# %%
## PHASE 6: Save All Necessary Artifacts for Web App
print("\n" + "=" * 80)
print("PHASE 6: Saving All Artifacts for Web Application")
print("=" * 80 + "\n")

# MODIFICATION: Use correct variable name `best_model_name_full`
original_best_model_name = best_model_name_full
optimized_model_name_for_app = f"{original_best_model_name}_Optimized"
print(f"Finalizing the OPTIMIZED ensemble model for deployment: '{optimized_model_name_for_app}'")

# MODIFICATION: ensure most_influential_base_model_name is defined.
# It is determined from ensemble SHAP in PART 1. If that fails, it might be None.
most_influential_base_model_name = most_influential_base_model_name_from_ensemble

artifacts = {
    'final_base_models_and_processors': final_base_models_and_processors,
    'optimized_meta_feature_names': optimized_meta_feature_names,
    'final_model_feature_columns': optimized_feature_names,
    'other_features_names': other_features_names,
    'file_name': file_name,
    'best_model_name': optimized_model_name_for_app,
    'final_ensemble_model': final_optimized_ensemble_model,
    'optimization_bounds': bounds,
    # MODIFICATION: Pass the SHAP results from the base model analysis in PART 2
    'most_influential_base_model_name': part2_analyzed_model_name,  # The model actually analyzed
    'base_model_shap_values': part2_shap_values,
    'base_model_all_feature_names': part2_all_feature_names,
    'base_model_fp_calculator': part2_fp_calculator_instance,
    'base_model_input_mols': part2_input_mols_for_fp_calc,
}

artifact_path = os.path.join(base_results_path, 'webapp_artifacts_optimized.pkl')
with open(artifact_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\nAll necessary artifacts for the OPTIMIZED model have been saved to: {artifact_path}")
print("This file can now be used for a Flask/Streamlit web application.")
# %% md
## PHASE 7: Detailed SHAP Analysis for a Specific Pollutant
# %%
# =============================================================================
# 导入必要的库
# =============================================================================
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
from typing import List, Optional, Tuple

warnings.filterwarnings('ignore')


# =============================================================================
# 核心功能函数 (无修改)
# =============================================================================
def calculate_atom_shap_weights(
        smiles_to_find: str,
        all_input_mols: List[Chem.Mol],
        all_shap_values: np.ndarray,
        all_feature_names: List[str],
        fp_calculator,
        model_name: str
) -> Tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
    """
    在数据集中找到指定分子，并计算其每个原子的SHAP贡献权重。
    """
    try:
        target_mol_rdkit = Chem.MolFromSmiles(smiles_to_find)
        target_smiles_canonical = Chem.MolToSmiles(target_mol_rdkit)
        all_smiles_canonical = [Chem.MolToSmiles(m) for m in all_input_mols]

        analysis_idx = all_smiles_canonical.index(target_smiles_canonical)
        print(f"在分析数据集中找到分子，索引为 {analysis_idx}。")

        mol = all_input_mols[analysis_idx]
        shap_values_instance = all_shap_values[analysis_idx]

        # MODIFICATION: Since base models only have FP features, this check is simpler.
        # We can assume all features passed in all_feature_names belong to the model.
        fp_feature_indices = [i for i, name in enumerate(all_feature_names) if name.startswith(model_name)]
        if not fp_feature_indices:  # If names don't match, assume all features are for this model
            fp_feature_indices = list(range(len(all_feature_names)))

        fp_shap_values = shap_values_instance[fp_feature_indices]
        fp_feature_names = [all_feature_names[i] for i in fp_feature_indices]

        instance_shap_df = pd.DataFrame({'feature_name': fp_feature_names, 'shap_value': fp_shap_values})
        instance_shap_df['bit_id'] = instance_shap_df['feature_name'].apply(lambda x: int(x.split('_')[-1]))

        try:
            # MODIFICATION: Handle new skfp attribute name `fp_size`
            radius = getattr(fp_calculator, 'radius', 2)
            nBits = getattr(fp_calculator, 'fp_size', getattr(fp_calculator, 'n_bits', 2048))
        except AttributeError:
            print("警告: 无法从fp_calculator获取参数, 使用默认值 radius=2, nBits=2048。")
            radius, nBits = 2, 2048

        bit_info = {}
        fp_instance = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bit_info)

        on_bits = set(fp_instance.GetOnBits())
        instance_shap_data = instance_shap_df[instance_shap_df['bit_id'].isin(on_bits)]
        bit_to_shap = pd.Series(instance_shap_data.shap_value.values, index=instance_shap_data.bit_id).to_dict()

        atom_weights = np.zeros(mol.GetNumAtoms())

        for bit_id, atom_info_list in bit_info.items():
            if bit_id in bit_to_shap:
                shap_val = bit_to_shap[bit_id]
                involved_atoms = {atom_idx for atom_idx, _ in atom_info_list}
                for atom_idx in involved_atoms:
                    atom_weights[atom_idx] += shap_val

        return mol, atom_weights

    except ValueError:
        print(f"错误：SMILES字符串 '{smiles_to_find}' 在数据集中未找到。")
        return None, None
    except Exception as e:
        print(f"计算原子SHAP权重时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def read_cube_geometry(filepath: str) -> Optional[Tuple[List[int], np.ndarray]]:
    """
    从一个 .cube 文件中解析原子数、原子序数和坐标。(无修改)
    """
    try:
        with open(filepath, 'r') as f:
            f.readline()
            f.readline()
            line = f.readline().split()
            num_atoms = int(line[0])
            f.readline()
            f.readline()
            f.readline()
            atom_numbers = []
            coordinates = []
            for _ in range(num_atoms):
                atom_line = f.readline().split()
                atom_numbers.append(int(atom_line[0]))
                coordinates.append([float(c) for c in atom_line[2:5]])
        print(f"从 '{filepath}' 成功读取 {num_atoms} 个原子的坐标。")
        return atom_numbers, np.array(coordinates)
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"错误: 无法读取或解析参考cube文件 '{filepath}': {e}")
        return None


def apply_new_coordinates(mol: Chem.Mol, ref_atom_numbers: List[int], ref_coords: np.ndarray) -> Optional[Chem.Mol]:
    """
    将一组外部坐标应用到 RDKit 分子对象上。(无修改)
    """
    mol_atom_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    if mol.GetNumAtoms() != len(ref_atom_numbers):
        print(f"错误: 原子数量不匹配！SMILES分子有 {mol.GetNumAtoms()} 个原子，"
              f"而参考文件有 {len(ref_atom_numbers)} 个原子。")
        return None
    if sorted(mol_atom_numbers) != sorted(ref_atom_numbers):
        print("错误: 原子类型不匹配！SMILES与参考文件的化学式不同。")
        return None
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        pos = ref_coords[i]
        conformer.SetAtomPosition(i, pos)
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    print("已成功将参考坐标应用到分子对象。")
    return mol


def generate_3d_conformer(mol: Chem.Mol) -> Chem.Mol:
    """生成3D构象 (无修改)"""
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        print("警告: MMFF优化失败，使用初步嵌入的结构。")
    return mol_3d


def export_for_avogadro_mol2(mol_3d: Chem.Mol, atom_weights: np.ndarray, filepath: str):
    """导出 .mol2 文件 (无修改)"""
    print("导出 .mol2 文件...")
    for i, atom in enumerate(mol_3d.GetAtoms()):
        weight = atom_weights[i] if i < len(atom_weights) else 0.0
        atom.SetDoubleProp('_TriposPartialCharge', weight)
    Chem.MolToMol2File(mol_3d, filepath)
    print(f"成功保存到: {filepath}")


# =============================================================================
# ★★★ 新增函数：导出原子权重到 TXT 文件 ★★★
# =============================================================================
def export_atom_weights_to_txt(mol_3d: Chem.Mol, atom_weights: np.ndarray, filepath: str):
    """
    将每个原子的SHAP权重导出到文本文件。
    """
    print(f"正在导出原子SHAP权重到: {filepath}")
    with open(filepath, 'w') as f:
        f.write("# Atom_Index   Atom_Symbol   SHAP_Weight\n")
        f.write("# ======================================\n")
        for atom in mol_3d.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            weight = atom_weights[idx] if idx < len(atom_weights) else 0.0
            f.write(f"{idx:<12d}{symbol:<13s}{weight:e}\n")
    print("成功导出权重文件。")


def export_for_avogadro_cube(
        mol_3d: Chem.Mol,
        atom_weights: np.ndarray,
        filepath: str,
        grid_spacing: float = 0.2,
        sigma: float = 1.0,
        normalize: bool = True
):
    """
    计算三维SHAP场并导出 .cube 文件。
    """
    print("计算三维SHAP场并导出 .cube 文件...")

    if normalize:
        print("信息: 正在对原子权重进行归一化处理。")
        max_abs_weight = np.max(np.abs(atom_weights))
        if max_abs_weight > 1e-9:
            print(f"归一化前最大绝对值为: {max_abs_weight:.4f}")
            atom_weights = atom_weights / max_abs_weight
        else:
            print("权重值过小，跳过归一化。")
    else:
        print("信息: 已跳过归一化，使用原始SHAP权重。")

    conf = mol_3d.GetConformer()
    atom_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_3d.GetNumAtoms())])

    min_coords = atom_coords.min(axis=0) - 5.0
    max_coords = atom_coords.max(axis=0) + 5.0
    grid_x = np.arange(min_coords[0], max_coords[0], grid_spacing)
    grid_y = np.arange(min_coords[1], max_coords[1], grid_spacing)
    grid_z = np.arange(min_coords[2], max_coords[2], grid_spacing)
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)

    grid_points_x, grid_points_y, grid_points_z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid_points = np.vstack([grid_points_x.ravel(), grid_points_y.ravel(), grid_points_z.ravel()]).T

    volumetric_data = np.zeros(grid_points.shape[0])
    for i, coord in enumerate(atom_coords):
        weight = atom_weights[i] if i < len(atom_weights) else 0.0
        if abs(weight) > 1e-9:
            dist_sq = np.sum((grid_points - coord) ** 2, axis=1)
            volumetric_data += weight * np.exp(-dist_sq / (2 * sigma ** 2))

    volumetric_data = volumetric_data.reshape((nx, ny, nz))

    with open(filepath, 'w') as f:
        f.write("SHAP contribution map\n")
        f.write("Generated from QSAR model (RAW, Non-Normalized Values)\n")
        f.write(f"{mol_3d.GetNumAtoms():5d} {min_coords[0]:12.6f} {min_coords[1]:12.6f} {min_coords[2]:12.6f}\n")
        f.write(f"{nx:5d} {grid_spacing:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {grid_spacing:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {grid_spacing:12.6f}\n")

        for i, atom in enumerate(mol_3d.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetAtomicNum():5d} {0.0:12.6f} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{volumetric_data[i, j, k]:13.5e} ")
                    if (k + 1) % 6 == 0 and k < nz - 1:
                        f.write("\n")
                f.write("\n")

    print(f"成功保存到: {filepath}")


# =============================================================================
# ★★★ 主控制函数修改 ★★★
# =============================================================================
def export_shap_for_avogadro(
        smiles_to_analyze: str,
        export_method: str,
        # MODIFICATION: Use the specific variables from the SHAP analysis of the target base model
        shap_values_base,
        input_mols_for_fp,
        all_feature_names_base,
        fp_calculator_base,
        target_base_model_name,  # Name of the model being analyzed (e.g., 'ECFP')
        base_models_path,
        reference_cube_file: Optional[str] = None,
        use_normalization: bool = True,
        export_weights_to_txt_file: bool = False  # ★★★ 新增控制参数 ★★★
):
    print(f"\n--- 开始为SMILES生成Avogadro文件: {smiles_to_analyze} ---")
    print(f"--- 使用方法: {export_method.upper()} ---")

    mol, atom_weights_no_h = calculate_atom_shap_weights(
        smiles_to_analyze, input_mols_for_fp, shap_values_base.values,
        all_feature_names_base, fp_calculator_base, target_base_model_name
    )
    if mol is None: return

    mol_3d = None
    if reference_cube_file:
        print(f"\n正在从参考文件加载3D坐标: {reference_cube_file}")
        geom_data = read_cube_geometry(reference_cube_file)
        if geom_data:
            ref_atom_nums, ref_coords = geom_data
            mol_with_h = Chem.AddHs(mol)
            mol_3d = apply_new_coordinates(mol_with_h, ref_atom_nums, ref_coords)
            if mol_3d is None:
                print("错误：无法将参考坐标应用到分子上，程序中止。")
                return

    if mol_3d is None:
        print("\n未提供或无法使用参考文件，正在生成新的3D分子构象...")
        mol_3d = generate_3d_conformer(mol)

    atom_weights_with_h = np.zeros(mol_3d.GetNumAtoms())
    original_indices = [a.GetIdx() for a in mol.GetAtoms()]
    for i, orig_idx in enumerate(original_indices):
        if i < len(atom_weights_no_h):
            atom_weights_with_h[orig_idx] = atom_weights_no_h[i]

    output_path = os.path.join(base_models_path, target_base_model_name, 'shap_analysis_influential')
    os.makedirs(output_path, exist_ok=True)
    sanitized_smiles = "".join(c for c in smiles_to_analyze if c.isalnum())[:30]

    # ★★★ 新增逻辑：如果需要，则导出权重到TXT文件 ★★★
    if export_weights_to_txt_file:
        txt_filepath = os.path.join(output_path, f'shap_weights_{sanitized_smiles}.txt')
        # 使用原始权重（未归一化）进行导出
        export_atom_weights_to_txt(mol_3d, atom_weights_with_h, txt_filepath)

    if export_method.lower() == 'mol2':
        filepath = os.path.join(output_path, f'shap_{sanitized_smiles}_aligned.mol2')
        export_for_avogadro_mol2(mol_3d, atom_weights_with_h, filepath)
    elif export_method.lower() == 'cube':
        suffix = "" if use_normalization else "_nonorm"
        filepath = os.path.join(output_path, f'shap_{sanitized_smiles}_aligned{suffix}.cube')
        export_for_avogadro_cube(mol_3d, atom_weights_with_h, filepath,
                                 grid_spacing=0.2, sigma=1.0, normalize=use_normalization)
    else:
        print(f"错误: 未知的导出方法 '{export_method}'。请选择 'mol2' 或 'cube'。")

    print("\n--- 导出完成。 ---")


# =============================================================================
# PHASE 7: AVOGADRO VISUALIZATION - 执行部分
# =============================================================================
# --- 使用从前面步骤加载的真实数据 ---
try:
    if 'part2_shap_values' not in locals() or part2_shap_values is None:
        raise NameError("SHAP analysis results from Part 2 not found. Cannot proceed with Phase 7.")
    print("PHASE 5/PART 2中的SHAP分析数据已加载，可用于PHASE 7。")

    # =============================================================================
    # ★★★ 实际执行：在这里控制导出选项 ★★★
    # =============================================================================
    SMILES_to_analyze = 'CC(C)(C(=O)O)OC1=CC=C(C=C1)Cl'
    EXPORT_METHOD = 'cube'
    # MODIFICATION: This path is specific to your machine, ensure it is correct.
    # REFERENCE_FILE_PATH = 'C:/Users/32928/Desktop/其他/Luo-ML-20250422/data/redox/Ensemble_Models_Results_20250613_113403/Base_Models_XGB/ECFP/mol_5376_LUMO.txt'
    REFERENCE_FILE_PATH = None  # Set to a valid path or None to generate conformer

    # --- 控制开关 ---
    USE_NORMALIZATION = False
    EXPORT_SHAP_WEIGHTS_TO_TXT = True  # ★★★ 设置为 True 来导出TXT文件 ★★★

    # 检查参考文件是否存在
    if REFERENCE_FILE_PATH and not os.path.exists(REFERENCE_FILE_PATH):
        print(f"警告：参考文件 '{REFERENCE_FILE_PATH}' 不存在。将生成新的3D坐标。")
        REFERENCE_FILE_PATH = None

    export_shap_for_avogadro(
        smiles_to_analyze=SMILES_to_analyze,
        export_method=EXPORT_METHOD,
        # MODIFICATION: Pass the correct variables from the Part 2 analysis
        shap_values_base=part2_shap_values,
        input_mols_for_fp=part2_input_mols_for_fp_calc,
        all_feature_names_base=part2_all_feature_names,
        fp_calculator_base=part2_fp_calculator_instance,
        target_base_model_name=part2_analyzed_model_name,
        base_models_path=base_models_path,
        reference_cube_file=REFERENCE_FILE_PATH,
        use_normalization=USE_NORMALIZATION,
        export_weights_to_txt_file=EXPORT_SHAP_WEIGHTS_TO_TXT  # <-- 将新开关传递给主函数
    )

except NameError as e:
    print(f"跳过 PHASE 7: {e}")
    print("此阶段依赖于PHASE 5中对目标基础模型（如ECFP）的成功SHAP分析。如果PHASE 5失败或跳过了该部分，则无法执行此阶段。")
# %%

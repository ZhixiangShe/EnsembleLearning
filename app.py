import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="QSAR/QSPR Intelligent Platform", layout="centered")

# ==========================================
# 1. 加载预训练模型
# ==========================================
@st.cache_resource(show_spinner="Loading model files...")
def load_model_assets(dataset_name):
    file_map = {
        "O3": "model_assets_O3.pkl",
        "ZVI": "model_assets_ZVI.pkl"
    }
    
    path_in_folder = os.path.join("deploy_models", file_map[dataset_name])
    path_in_root = file_map[dataset_name]
    
    target_path = None
    if os.path.exists(path_in_folder):
        target_path = path_in_folder
    elif os.path.exists(path_in_root):
        target_path = path_in_root
        
    if target_path is None:
        st.error(f"❌ Cannot find model file '{file_map[dataset_name]}'.")
        return None
        
    try:
        with open(target_path, 'rb') as f:
            assets = pickle.load(f)
        return assets
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__} - {str(e)}")
        return None

# ==========================================
# 2. 交互式预测核心函数 (预训练模型)
# ==========================================
def predict_smiles(smiles, env_values_dict, assets):
    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None:
        return None, "Invalid SMILES"
    
    try:
        mol_3d = Chem.AddHs(mol_2d)
        embed_res = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if embed_res == -1:
            return None, "Failed to generate 3D structure for this SMILES."
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_3d.conf_id = 0
        if hasattr(mol_3d, 'SetIntProp'):
            mol_3d.SetIntProp("conf_id", 0)
    except Exception as e:
        return None, f"Error generating 3D conformer: {str(e)}"
    
    base_preds = []
    top_fp_names = assets['top_fp_names']
    
    for fp_name in top_fp_names:
        processor = assets['base_models_processors'][fp_name]
        try:
            fp_raw = processor['fp_calculator'].transform([mol_3d])
            fp = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw)
            fp_processed = processor['imputer'].transform(processor['scaler'].transform(fp))
            pred = processor['model'].predict(fp_processed)[0]
            base_preds.append(pred)
        except Exception as e:
            return None, f"Error calculating {fp_name}: {str(e)}"
    
    env_features = [env_values_dict.get(name, 0.0) for name in assets['env_feature_names']]
    meta_input = np.array(base_preds + env_features).reshape(1, -1)
    
    final_pred = assets['meta_model'].predict(meta_input)[0]
    return final_pred, mol_2d

# ==========================================
# 3. UI 布局
# ==========================================
st.title("QSAR/QSPR Intelligent Modeling & Prediction Platform")
st.markdown("A Visual Machine Learning Workflow for QSAR/QSPR Studies in Chemistry")
st.divider()

# ------------- Module 1: Project Configuration -------------
st.header("Module 1: Built-in Model Configuration")
dataset_choice = st.radio("Select Pre-trained Dataset Model", ["O3", "ZVI"], horizontal=True)
assets = load_model_assets(dataset_choice)
st.divider()

# ------------- Module 2 & 3: Interactive Prediction & SHAP -------------
if assets is not None:
    st.header("Module 2: Interactive Analysis & Prediction")
    target_smiles = st.text_input("Enter Target Molecule SMILES", "CC(C)(C)C1=NN=C(S1)NC(=O)NC")
    
    env_inputs = {}
    if len(assets['env_feature_names']) > 0:
        st.write("Environmental Variables:")
        cols = st.columns(len(assets['env_feature_names']))
        for i, env_name in enumerate(assets['env_feature_names']):
            env_inputs[env_name] = cols[i].number_input(env_name, value=7.0 if env_name.lower() == 'ph' else 25.0)
            
    if st.button("Predict Kob Value", type="primary"):
        with st.spinner("Generating 3D conformer & Calculating..."):
            pred_value, result_mol = predict_smiles(target_smiles, env_inputs, assets)
            if pred_value is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    img = Draw.MolToImage(result_mol, size=(300, 300))
                    st.image(img, caption="2D Molecular Structure")
                with c2:
                    st.success("Prediction Complete!")
                    st.metric(label="Predicted Kob Value", value=f"{pred_value:.4f}")
            else:
                st.error(result_mol)

    st.divider()

    st.header("Module 3: Molecular Interpretability (Atomic SHAP)")
    shap_smiles = st.text_input("Enter SMILES for Interpretability", target_smiles, key="shap_input")
    
    if st.button("Generate Atomic Contribution Plot"):
        mol_shap = Chem.MolFromSmiles(shap_smiles)
        if mol_shap:
            with st.spinner("Calculating ECFP Atomic Contributions..."):
                try:
                    if 'ecfp_surrogate' not in assets:
                        st.warning("⚠️ ECFP Surrogate model not found in the .pkl file.")
                    else:
                        ecfp_model = assets['ecfp_surrogate']
                        def get_pred_for_shap(fp_vect):
                            fp_array = np.array(fp_vect).reshape(1, -1)
                            return float(ecfp_model.predict(fp_array)[0])
                        
                        weights = SimilarityMaps.GetAtomicWeightsForModel(
                            mol_shap, 
                            lambda m, i: SimilarityMaps.GetMorganFingerprint(m, atomId=i, radius=2, nBits=2048), 
                            get_pred_for_shap
                        )
                        
                        d2d = rdMolDraw2D.MolDraw2DSVG(450, 450)
                        SimilarityMaps.GetSimilarityMapFromWeights(mol_shap, weights, colorMap='coolwarm', draw2d=d2d)
                        d2d.FinishDrawing()
                        
                        components.html(f"<div style='text-align: center;'>{d2d.GetDrawingText()}</div>", width=600, height=480)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
        else:
            st.error("Invalid SMILES.")

st.divider()

# ==========================================
# 4. Module 4: Custom Stacking 5-Fold CV Model
# ==========================================
st.header("Module 4: Train Custom Stacking Model")
st.markdown("""
Upload your own CSV to build an **XGBoost Stacking Model (5-Fold CV)** using Multiple Fingerprints.
⚠️ *Note: 3D Fingerprints (RDF, MORSE) require 3D conformer generation which is computationally expensive. Datasets > 500 molecules may cause server timeout.*
""")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

# 指纹计算字典定义
FP_CALCULATORS = {
    "Morgan (2D)": lambda m: np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)),
    "MACCS (2D)": lambda m: np.array(rdMolDescriptors.GetMACCSKeysFingerprint(m)),
    "RDF (3D)": lambda m: np.array(rdMolDescriptors.CalcRDF(m)),
    "MORSE (3D)": lambda m: np.array(rdMolDescriptors.CalcMORSE(m)),
    "AUTOCORR3D (3D)": lambda m: np.array(rdMolDescriptors.CalcAUTOCORR3D(m))
}

if uploaded_file is not None:
    try:
        df_custom = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df_custom.head(3))
        all_cols = df_custom.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            smiles_col = st.selectbox("Select SMILES column", all_cols, index=0)
        with col2:
            target_col = st.selectbox("Select Target (y) column", all_cols, index=min(1, len(all_cols)-1))
            
        remaining_cols = [c for c in all_cols if c not in [smiles_col, target_col]]
        env_cols = st.multiselect("Select Environmental Variables (Optional)", remaining_cols)
        
        selected_fps = st.multiselect(
            "Select Base Fingerprints for Stacking", 
            list(FP_CALCULATORS.keys()), 
            default=["Morgan (2D)", "MACCS (2D)", "RDF (3D)"]
        )
        
        if st.button("🚀 Run 5-Fold CV Stacking Pipeline", type="primary"):
            if not selected_fps:
                st.error("Please select at least one fingerprint.")
            else:
                with st.spinner("Processing Pipeline... This may take a while depending on data size."):
                    # 判断是否需要生成 3D 构象
                    need_3d = any("3D" in fp for fp in selected_fps)
                    
                    df_clean = df_custom.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
                    
                    valid_indices = []
                    computed_mols = []
                    
                    st.write(f"Step 1: Generating {'3D' if need_3d else '2D'} conformers...")
                    progress_bar = st.progress(0)
                    
                    for i, row in df_clean.iterrows():
                        smi = str(row[smiles_col])
                        mol = Chem.MolFromSmiles(smi)
                        if mol is not None:
                            try:
                                if need_3d:
                                    mol = Chem.AddHs(mol)
                                    embed_res = AllChem.EmbedMolecule(mol, randomSeed=42)
                                    if embed_res != -1:
                                        AllChem.MMFFOptimizeMolecule(mol)
                                        mol.conf_id = 0
                                        computed_mols.append(mol)
                                        valid_indices.append(i)
                                else:
                                    computed_mols.append(mol)
                                    valid_indices.append(i)
                            except:
                                pass # 跳过失败的分子
                        
                        if i % 10 == 0 or i == len(df_clean)-1:
                            progress_bar.progress(min(1.0, (i + 1) / len(df_clean)))
                    
                    if len(valid_indices) == 0:
                        st.error("No valid molecules could be processed.")
                        st.stop()
                        
                    st.success(f"Successfully processed {len(valid_indices)} molecules.")
                    
                    # 准备目标变量 y 和 环境特征 X_env
                    y = df_clean.iloc[valid_indices][target_col].values
                    X_env = None
                    if len(env_cols) > 0:
                        X_env = df_clean.iloc[valid_indices][env_cols].values
                        X_env = np.nan_to_num(X_env, nan=np.nanmean(X_env, axis=0))
                    
                    # 划分总体训练集和测试集 (80% 训练, 20% 测试)
                    indices = np.arange(len(valid_indices))
                    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
                    
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    
                    # ----------------- Stacking 核心流程 -----------------
                    st.write("Step 2: Calculating Fingerprints & 5-Fold Cross-Validation for Base Models...")
                    
                    # 初始化存储 Meta-features 的矩阵
                    meta_X_train_base = np.zeros((len(train_idx), len(selected_fps)))
                    meta_X_test_base = np.zeros((len(test_idx), len(selected_fps)))
                    
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    
                    for fp_idx, fp_name in enumerate(selected_fps):
                        st.text(f"→ Training Base Model: XGBoost with {fp_name}")
                        
                        # 1. 计算当前指纹
                        calc_func = FP_CALCULATORS[fp_name]
                        fp_data = [calc_func(mol) for mol in computed_mols]
                        X_fp = np.array(fp_data)
                        
                        # 2. 数据标准化
                        scaler = StandardScaler()
                        X_fp_scaled = scaler.fit_transform(X_fp)
                        
                        X_train_fp = X_fp_scaled[train_idx]
                        X_test_fp = X_fp_scaled[test_idx]
                        
                        # 固定超参数的基础模型
                        base_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
                        
                        # 3. 5-Fold OOF 预测 (构成训练集的 Meta-feature)
                        oof_preds = cross_val_predict(base_model, X_train_fp, y_train, cv=kf)
                        meta_X_train_base[:, fp_idx] = oof_preds
                        
                        # 4. 在全部训练集上拟合，预测测试集 (构成测试集的 Meta-feature)
                        base_model.fit(X_train_fp, y_train)
                        meta_X_test_base[:, fp_idx] = base_model.predict(X_test_fp)

                    # ----------------- Meta-Model 训练 -----------------
                    st.write("Step 3: Training Meta-Model (Stacking)...")
                    
                    # 拼接环境特征
                    if X_env is not None:
                        env_train = X_env[train_idx]
                        env_test = X_env[test_idx]
                        Meta_X_train = np.hstack((meta_X_train_base, env_train))
                        Meta_X_test = np.hstack((meta_X_test_base, env_test))
                    else:
                        Meta_X_train = meta_X_train_base
                        Meta_X_test = meta_X_test_base
                    
                    # 顶层 Meta-Model
                    meta_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
                    meta_model.fit(Meta_X_train, y_train)
                    
                    # 最终预测
                    y_pred_train = meta_model.predict(Meta_X_train)
                    y_pred_test = meta_model.predict(Meta_X_test)
                    
                    # 计算评估指标
                    r2_train = r2_score(y_train, y_pred_train)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    r2_test = r2_score(y_test, y_pred_test)
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    
                    # ----------------- 展示结果 -----------------
                    st.success("Pipeline Execution Finished!")
                    
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Meta Train R²", f"{r2_train:.3f}")
                    metrics_cols[1].metric("Meta Test R²", f"{r2_test:.3f}")
                    metrics_cols[2].metric("Meta Test RMSE", f"{rmse_test:.3f}")
                    metrics_cols[3].metric("Meta Test MAE", f"{mae_test:.3f}")
                    
                    # 绘制预测散点图
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_train, y_pred_train, alpha=0.5, label='Train (OOF Meta)', color='#4C72B0')
                    ax.scatter(y_test, y_pred_test, alpha=0.7, label='Test', color='#DD8452')
                    
                    min_val = min(min(y), min(y_pred_train), min(y_pred_test))
                    max_val = max(max(y), max(y_pred_train), max(y_pred_test))
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
                    
                    ax.set_xlabel(f"Actual {target_col}")
                    ax.set_ylabel(f"Predicted {target_col}")
                    ax.set_title("Stacking Ensemble: Actual vs Predicted")
                    ax.legend()
                    ax.grid(True, linestyle=':', alpha=0.7)
                    
                    st.pyplot(fig)
                    
    except Exception as e:
        st.error(f"Error during execution: {str(e)}")

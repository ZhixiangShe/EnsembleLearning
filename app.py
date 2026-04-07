import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="Ensemble Learning Platform for Advanced Treatments", layout="centered")

# ==========================================
# 1. 加载预训练模型 (模块 1)
# ==========================================
@st.cache_resource(show_spinner="Loading model files...")
def load_model_assets(dataset_name):
    file_map = {"O3": "model_assets_O3.pkl", "ZVI": "model_assets_ZVI.pkl"}
    path_in_folder = os.path.join("deploy_models", file_map[dataset_name])
    path_in_root = file_map[dataset_name]
    
    target_path = path_in_folder if os.path.exists(path_in_folder) else path_in_root if os.path.exists(path_in_root) else None
        
    if target_path is None: return None
    try:
        with open(target_path, 'rb') as f: return pickle.load(f)
    except: return None

# ==========================================
# 2. 预训练模型预测核心
# ==========================================
def predict_smiles(smiles, env_values_dict, assets):
    mol_2d = Chem.MolFromSmiles(smiles)
    if mol_2d is None: return None, "Invalid SMILES"
    try:
        mol_3d = Chem.AddHs(mol_2d)
        if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != -1:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_3d.conf_id = 0
        if hasattr(mol_3d, 'SetIntProp'): mol_3d.SetIntProp("conf_id", 0)
    except: pass
    
    base_preds = []
    for fp_name in assets['top_fp_names']:
        processor = assets['base_models_processors'][fp_name]
        try:
            fp_raw = processor['fp_calculator'].transform([mol_3d])
            fp = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw)
            fp_processed = processor['imputer'].transform(processor['scaler'].transform(fp))
            base_preds.append(processor['model'].predict(fp_processed)[0])
        except Exception as e: return None, str(e)
    
    env_features = [env_values_dict.get(name, 0.0) for name in assets['env_feature_names']]
    meta_input = np.array(base_preds + env_features).reshape(1, -1)
    return assets['meta_model'].predict(meta_input)[0], mol_2d

# ==========================================
# 3. UI 布局 (模块 1-3)
# ==========================================
st.title("Ensemble Learning Platform for Advanced Treatments")
st.markdown("An Ensemble Learning Framework for Pollutant Reactivity Prediction")
st.divider()

st.header("Module 1: Built-in Model Configuration")
dataset_choice = st.radio("Select Pre-trained Dataset Model", ["O3", "ZVI"], horizontal=True)
assets = load_model_assets(dataset_choice)
st.divider()

if assets is not None:
    st.header("Module 2: Interactive Analysis & Prediction")
    target_smiles = st.text_input("Enter Target Molecule SMILES", "CC(C)(C)C1=NN=C(S1)NC(=O)NC")
    env_inputs = {}
    if len(assets['env_feature_names']) > 0:
        cols = st.columns(len(assets['env_feature_names']))
        for i, env_name in enumerate(assets['env_feature_names']):
            env_inputs[env_name] = cols[i].number_input(env_name, value=7.0 if env_name.lower() == 'ph' else 25.0)
            
    if st.button("Predict Kob Value", type="primary"):
        with st.spinner("Calculating..."):
            pred_value, result_mol = predict_smiles(target_smiles, env_inputs, assets)
            if pred_value is not None:
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(Draw.MolToImage(result_mol, size=(300, 300)), caption="2D Structure")
                with c2:
                    st.success("Prediction Complete!")
                    st.metric(label="Predicted Kob", value=f"{pred_value:.4f}")

    st.divider()

    st.header("Module 3: Molecular Interpretability (Atomic SHAP)")
    shap_smiles = st.text_input("Enter SMILES for SHAP", target_smiles, key="shap_input")
    if st.button("Generate Atomic Contribution Plot"):
        mol_shap = Chem.MolFromSmiles(shap_smiles)
        if mol_shap and 'ecfp_surrogate' in assets:
            with st.spinner("Calculating..."):
                def get_pred_for_shap(fp_vect): return float(assets['ecfp_surrogate'].predict(np.array(fp_vect).reshape(1, -1))[0])
                weights = SimilarityMaps.GetAtomicWeightsForModel(mol_shap, lambda m, i: SimilarityMaps.GetMorganFingerprint(m, atomId=i, radius=2, nBits=2048), get_pred_for_shap)
                d2d = rdMolDraw2D.MolDraw2DSVG(450, 450)
                SimilarityMaps.GetSimilarityMapFromWeights(mol_shap, weights, colorMap='coolwarm', draw2d=d2d)
                d2d.FinishDrawing()
                components.html(f"<div style='text-align: center;'>{d2d.GetDrawingText()}</div>", width=600, height=480)
st.divider()

# ==========================================
# 4. Module 4: 自动网格搜索热图 & Top3 Stacking 集成
# ==========================================
st.header("Module 4: AutoML Profiling & Stacking Ensemble")
st.markdown("""
This module automatically evaluates **6 Machine Learning algorithms** across **14 Molecular Fingerprints (2D & 3D)**.
1. It generates an interactive $R^2$ Heatmap.
2. It automatically selects the **Top 3 Models** (Model + FP combinations).
3. It builds a **5-Fold Stacking Ensemble** using the Top 3 base models.
""")

uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

# 提取器定义 (对齐所需指纹，使用 RDKit 原生确保网页端不崩)
FP_DICT = {
    "MQNs": lambda m: np.array(rdMolDescriptors.CalcMQNs(m)),
    "MACCS": lambda m: np.array(rdMolDescriptors.GetMACCSKeysFingerprint(m)),
    "PatternFP": lambda m: np.array(Chem.PatternFingerprint(m)),
    "AtomPair_2D": lambda m: np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048)),
    "TopologicalTorsion": lambda m: np.array(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=2048)),
    "ECFP": lambda m: np.array(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)),
    "RDKitFP": lambda m: np.array(Chem.RDKFingerprint(m)),
    "Autocorr_2D": lambda m: np.array(rdMolDescriptors.CalcAUTOCORR2D(m)),
    "MORSE_3D": lambda m: np.array(rdMolDescriptors.CalcMORSE(m)),
    "RDF_3D": lambda m: np.array(rdMolDescriptors.CalcRDF(m)),
    "USR_3D": lambda m: np.array(rdMolDescriptors.GetUSR(m)) if hasattr(rdMolDescriptors, 'GetUSR') else np.zeros(12),
}

# 基模型定义
MODELS_DICT = {
    "XGB": xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1),
    "RF": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "DT": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    "SVR": SVR(C=1.0, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
}

if uploaded_file is not None:
    df_custom = pd.read_csv(uploaded_file)
    cols = df_custom.columns.tolist()
    
    c1, c2 = st.columns(2)
    with c1: smi_col = st.selectbox("SMILES Column", cols, index=0)
    with c2: tgt_col = st.selectbox("Target Column", cols, index=min(1, len(cols)-1))
    
    env_cols = st.multiselect("Environmental Variables (Optional)", [c for c in cols if c not in [smi_col, tgt_col]])
    
    if st.button("🚀 Start Full AutoML Pipeline", type="primary"):
        df_clean = df_custom.dropna(subset=[smi_col, tgt_col]).reset_index(drop=True)
        
        # 1. 构象生成与准备
        st.write("Step 1/5: Generating 2D/3D Molecular Conformers...")
        mols, valid_idx = [], []
        pb = st.progress(0)
        
        for i, row in df_clean.iterrows():
            mol = Chem.MolFromSmiles(str(row[smi_col]))
            if mol:
                try:
                    mol = Chem.AddHs(mol)
                    if AllChem.EmbedMolecule(mol, randomSeed=42) != -1:
                        AllChem.MMFFOptimizeMolecule(mol)
                    mol.conf_id = 0
                    mols.append(mol)
                    valid_idx.append(i)
                except: pass
            if i % 10 == 0: pb.progress(min(1.0, i/len(df_clean)))
            
        y = df_clean.iloc[valid_idx][tgt_col].values
        X_env = df_clean.iloc[valid_idx][env_cols].values if env_cols else None
        if X_env is not None: X_env = np.nan_to_num(X_env, nan=np.nanmean(X_env, axis=0))
        
        # 划分为 80% 评估集(Train), 20% 测试集(Test)
        idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
        y_train, y_test = y[idx_train], y[idx_test]
        
        # 2. 遍历计算热图矩阵
        st.write("Step 2/5: Calculating Fingerprints & Evaluating Base Models...")
        r2_matrix = pd.DataFrame(index=MODELS_DICT.keys(), columns=FP_DICT.keys())
        results_list = []
        
        # 预先计算所有指纹
        fp_data_cache = {}
        for fp_name, func in FP_DICT.items():
            try:
                raw_fp = np.array([func(m) for m in mols])
                # 标准化非常重要，否则 SVR 和 MLP 表现极差
                fp_data_cache[fp_name] = StandardScaler().fit_transform(raw_fp)
            except Exception as e:
                fp_data_cache[fp_name] = None
        
        total_tasks = len(MODELS_DICT) * len(FP_DICT)
        task_count = 0
        pb_eval = st.progress(0)
        
        for m_name, model in MODELS_DICT.items():
            for f_name in FP_DICT.keys():
                X_all = fp_data_cache[f_name]
                if X_all is not None:
                    try:
                        # 使用训练集训练，在测试集上评估 (速度最快的方法)
                        X_tr, X_te = X_all[idx_train], X_all[idx_test]
                        model.fit(X_tr, y_train)
                        score = r2_score(y_test, model.predict(X_te))
                        r2_matrix.loc[m_name, f_name] = score
                        results_list.append((score, m_name, f_name))
                    except:
                        r2_matrix.loc[m_name, f_name] = -1.0
                task_count += 1
                pb_eval.progress(task_count / total_tasks)
                
        # 3. 绘制交互式热图 (Plotly)
        st.write("Step 3/5: Interactive Performance Heatmap (Test $R^2$)")
        r2_matrix = r2_matrix.astype(float).fillna(0.0)
        fig_heat = px.imshow(r2_matrix, text_auto=".2f", aspect="auto", 
                             color_continuous_scale="RdBu_r", 
                             title="Model vs Fingerprint Performance Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # 4. 提取 Top 3
        st.write("Step 4/5: Selecting Top 3 Combinations for Stacking...")
        results_list.sort(key=lambda x: x[0], reverse=True)
        top_3 = results_list[:3]
        
        for i, (score, m_name, f_name) in enumerate(top_3):
            st.success(f"🏅 Rank {i+1}: **{m_name}** with **{f_name}** (Base $R^2$: {score:.3f})")
            
        # 5. Stacking 5-Fold CV
        st.write("Step 5/5: Training Stacking Meta-Model (5-Fold CV)...")
        
        meta_X_train = np.zeros((len(y_train), 3))
        meta_X_test = np.zeros((len(y_test), 3))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, (score, m_name, f_name) in enumerate(top_3):
            base_model = MODELS_DICT[m_name]
            X_fp = fp_data_cache[f_name]
            X_tr, X_te = X_fp[idx_train], X_fp[idx_test]
            
            # OOF 预测生成训练集 Meta-特征
            meta_X_train[:, i] = cross_val_predict(base_model, X_tr, y_train, cv=kf)
            # 全量拟合后预测测试集 Meta-特征
            base_model.fit(X_tr, y_train)
            meta_X_test[:, i] = base_model.predict(X_te)
            
        # 拼接环境参数
        if X_env is not None:
            meta_X_train = np.hstack((meta_X_train, X_env[idx_train]))
            meta_X_test = np.hstack((meta_X_test, X_env[idx_test]))
            
        # 顶层 Meta 模型拟合
        meta_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        meta_model.fit(meta_X_train, y_train)
        
        y_pred_tr = meta_model.predict(meta_X_train)
        y_pred_te = meta_model.predict(meta_X_test)
        
        # 指标计算
        st.subheader("Final Stacking Ensemble Results")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
        m_c2.metric("Test R²", f"{r2_score(y_test, y_pred_te):.3f}")
        m_c3.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_te)):.3f}")
        m_c4.metric("Test MAE", f"{mean_absolute_error(y_test, y_pred_te):.3f}")
        
        # 最终散点拟合图 (Matplotlib)
        fig_scatter, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_train, y_pred_tr, alpha=0.5, label='Train (OOF)', color='#4C72B0')
        ax.scatter(y_test, y_pred_te, alpha=0.7, label='Test', color='#DD8452')
        min_v = min(min(y), min(y_pred_tr), min(y_pred_te))
        max_v = max(max(y), max(y_pred_tr), max(y_pred_te))
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Ideal')
        ax.set_xlabel(f"Actual {tgt_col}")
        ax.set_ylabel(f"Predicted {tgt_col}")
        ax.set_title("Stacking Ensemble: Actual vs Predicted")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
        st.pyplot(fig_scatter)

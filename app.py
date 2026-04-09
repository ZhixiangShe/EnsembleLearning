import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.express as px
import streamlit.components.v1 as components
import xgboost as xgb
from PIL import Image

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
# 1. 加载预训练模型 (模块 1) - 增加了详细错误捕捉
# ==========================================
@st.cache_resource(show_spinner="Loading model files...")
def load_model_assets(dataset_name):
    file_map = {"O3": "model_assets_O3.pkl", "ZVI": "model_assets_ZVI.pkl"}
    path_in_folder = os.path.join("deploy_models", file_map[dataset_name])
    path_in_root = file_map[dataset_name]
    
    target_path = path_in_folder if os.path.exists(path_in_folder) else path_in_root if os.path.exists(path_in_root) else None
        
    if target_path is None: 
        return None, "File not found"
        
    try:
        with open(target_path, 'rb') as f: 
            return pickle.load(f), "Success"
    except Exception as e:
        # 将真实的错误捕捉下来
        return None, f"Failed to load `{target_path}`. Error details: {str(e)}"

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
# 3. UI 布局 (模块 1-2)
# ==========================================
st.title("Ensemble Learning Platform for Advanced Treatments")
st.markdown("An Ensemble Learning Framework for Pollutant Reactivity Prediction")
st.divider()

st.header("Module 1: Built-in Model Configuration")
dataset_choice = st.radio("Select Pre-trained Dataset Model", ["O3", "ZVI"], horizontal=True)

# 获取模型和状态信息
assets, load_status = load_model_assets(dataset_choice)
st.divider()

# ==========================================
# 关键修改：显示具体的报错原因
# ==========================================
if assets is None:
    if load_status == "File not found":
        st.error(f"⚠️ Missing Model Files: Cannot find `model_assets_{dataset_choice}.pkl`.")
    else:
        st.error("⚠️ Model file exists, but failed to load!")
        st.error(f"**Error Message:** {load_status}")
        st.info("💡 **Diagnosis:** If you see `ModuleNotFoundError` or `UnpicklingError`, it means the versions of `scikit-learn` or `xgboost` in your `requirements.txt` are different from the versions on your local PC when you created the .pkl file. Please strictly match the library versions in `requirements.txt`.")
else:
    st.header("Module 2: Interactive Prediction")
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

    # ==========================================
    # 模块 3: 顶刊级分子解释性
    # ==========================================
    st.header("Module 3: Molecular Interpretability")
    shap_smiles = st.text_input("Enter SMILES for Interpretability Analysis", target_smiles, key="shap_input")
    
    if st.button("Generate Interpretability Plots"):
        mol_shap = Chem.MolFromSmiles(shap_smiles)
        if mol_shap and 'ecfp_surrogate' in assets:
            with st.spinner("Analyzing Atomic & Fragment Contributions..."):
                
                # --- 图 1: 高清原子贡献图 ---
                st.subheader("🔴🔵 Atomic Contribution Plot")
                def get_pred_for_shap(fp_vect): 
                    return float(assets['ecfp_surrogate'].predict(np.array(fp_vect).reshape(1, -1))[0])
                
                weights = SimilarityMaps.GetAtomicWeightsForModel(
                    mol_shap, 
                    lambda m, i: SimilarityMaps.GetMorganFingerprint(m, atomId=i, radius=2, nBits=2048), 
                    get_pred_for_shap
                )
                
                norm = mcolors.TwoSlopeNorm(vmin=min(weights)-1e-5, vcenter=0, vmax=max(weights)+1e-5)
                cmap = plt.get_cmap('bwr')
                
                atom_colors = {}
                for i, w in enumerate(weights):
                    if abs(w) > 1e-4:
                        atom_colors[i] = cmap(norm(w))[:3]
                        
                d2d = rdMolDraw2D.MolDraw2DSVG(500, 400)
                opts = d2d.drawOptions()
                opts.clearBackground = True
                opts.setBackgroundColour((1, 1, 1, 1))
                opts.highlightRadius = 0.4
                
                rdMolDraw2D.PrepareAndDrawMolecule(
                    d2d, mol_shap, 
                    highlightAtoms=list(atom_colors.keys()), 
                    highlightAtomColors=atom_colors
                )
                d2d.FinishDrawing()
                
                components.html(f"<div style='text-align: center;'>{d2d.GetDrawingText()}</div>", width=600, height=420)
                
                fig_cb, ax_cb = plt.subplots(figsize=(6, 0.6))
                fig_cb.subplots_adjust(bottom=0.5)
                cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb, orientation='horizontal')
                cb.set_label('SHAP Contribution (Negative vs Positive)', fontsize=10)
                st.pyplot(fig_cb)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # --- 图 2: 带有分子片段的柱状图 ---
                st.subheader("📊 Fragment Feature Importance")
                
                bit_info = {}
                fp = AllChem.GetMorganFingerprintAsBitVect(mol_shap, radius=2, nBits=2048, bitInfo=bit_info)
                active_bits = list(bit_info.keys())
                
                base_pred = get_pred_for_shap(fp)
                bit_contributions = {}
                fp_arr = np.array(fp)
                
                for bit in active_bits:
                    fp_mutated = fp_arr.copy()
                    fp_mutated[bit] = 0
                    mutated_pred = get_pred_for_shap(fp_mutated)
                    contrib = base_pred - mutated_pred
                    if abs(contrib) > 1e-4:
                        bit_contributions[bit] = contrib
                    
                top_bits = sorted(bit_contributions.keys(), key=lambda x: abs(bit_contributions[x]), reverse=True)[:6]
                
                if len(top_bits) > 0:
                    top_vals = [bit_contributions[b] for b in top_bits]
                    top_labels = [str(b) for b in top_bits]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(top_labels, top_vals, color='#5D98C8', width=0.4, edgecolor='white')
                    
                    y_max = max(max(top_vals), 0)
                    y_min = min(min(top_vals), 0)
                    y_range = y_max - y_min if (y_max - y_min) > 0 else 0.1
                    
                    ax.set_ylim(y_min - y_range * 0.5, y_max + y_range * 0.5)
                    
                    for i, bit in enumerate(top_bits):
                        try:
                            img = Draw.DrawMorganBit(mol_shap, int(bit), bit_info, useSVG=False)
                            if img is not None:
                                img_rgb = img.convert('RGB')
                                img_arr = np.array(img_rgb)
                                
                                imagebox = OffsetImage(img_arr, zoom=0.6)
                                sign = 1 if top_vals[i] >= 0 else -1
                                offset = y_range * 0.15
                                y_pos = top_vals[i] + (sign * offset)
                                
                                ab = AnnotationBbox(
                                    imagebox, (i, y_pos), 
                                    frameon=True, 
                                    bboxprops=dict(edgecolor='gray', boxstyle='round,pad=0.2', facecolor='white', alpha=0.9)
                                )
                                ax.add_artist(ab)
                        except Exception as e:
                            pass
                            
                    ax.set_ylabel("Feature Importance (Contribution)")
                    ax.set_xlabel("ECFP Feature (Bit ID)")
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
                    ax.grid(axis='y', linestyle=':', alpha=0.7)
                    
                    st.pyplot(fig)
                else:
                    st.info("Molecule is too simple or contributions are negligible.")

st.divider()

# ==========================================
# 4. Module 4: 自动网格搜索热图 & Top3 Stacking 集成
# ==========================================
st.header("Module 4: Ensemble Learning for New Data")
st.markdown("Upload your dataset, select algorithms & features, and build a custom Stacking model.")

uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

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
    
    st.markdown("#### 1. Data Mapping")
    c1, c2 = st.columns(2)
    with c1: smi_col = st.selectbox("SMILES Column", cols, index=0)
    with c2: tgt_col = st.selectbox("Target Column", cols, index=min(1, len(cols)-1))
    
    env_cols = st.multiselect("Environmental Variables (Optional)", [c for c in cols if c not in [smi_col, tgt_col]])
    
    st.markdown("#### 2. Select Algorithms and Features")
    sel_c1, sel_c2 = st.columns(2)
    with sel_c1:
        selected_models = st.multiselect(
            "Select Base Models", 
            options=list(MODELS_DICT.keys()), 
            default=["XGB", "RF", "KNN", "SVR"]
        )
    with sel_c2:
        selected_fps = st.multiselect(
            "Select Fingerprints (Features)", 
            options=list(FP_DICT.keys()), 
            default=["MACCS", "ECFP", "AtomPair_2D", "RDF_3D"]
        )
    
    if st.button("🚀 Start Full AutoML Pipeline", type="primary"):
        if not selected_models or not selected_fps:
            st.error("⚠️ Please select at least one Base Model and one Fingerprint!")
        else:
            df_clean = df_custom.dropna(subset=[smi_col, tgt_col]).reset_index(drop=True)
            need_3d = any("_3D" in fp for fp in selected_fps)
            
            st.write(f"Step 1/5: Generating Molecular Conformers (3D Optimization: {need_3d})...")
            mols, valid_idx = [], []
            pb = st.progress(0)
            
            for i, row in df_clean.iterrows():
                mol = Chem.MolFromSmiles(str(row[smi_col]))
                if mol:
                    try:
                        if need_3d:
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
            
            idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
            y_train, y_test = y[idx_train], y[idx_test]
            
            st.write("Step 2/5: Calculating Features & Evaluating Combinations...")
            r2_matrix = pd.DataFrame(index=selected_models, columns=selected_fps)
            results_list = []
            
            fp_data_cache = {}
            for fp_name in selected_fps:
                func = FP_DICT[fp_name]
                try:
                    raw_fp = np.array([func(m) for m in mols])
                    fp_data_cache[fp_name] = StandardScaler().fit_transform(raw_fp)
                except Exception as e:
                    fp_data_cache[fp_name] = None
            
            total_tasks = len(selected_models) * len(selected_fps)
            task_count = 0
            pb_eval = st.progress(0)
            
            for m_name in selected_models:
                model = MODELS_DICT[m_name]
                for f_name in selected_fps:
                    X_all = fp_data_cache[f_name]
                    if X_all is not None:
                        try:
                            X_tr, X_te = X_all[idx_train], X_all[idx_test]
                            model.fit(X_tr, y_train)
                            score = r2_score(y_test, model.predict(X_te))
                            r2_matrix.loc[m_name, f_name] = score
                            results_list.append((score, m_name, f_name))
                        except:
                            r2_matrix.loc[m_name, f_name] = -1.0
                    task_count += 1
                    pb_eval.progress(task_count / total_tasks)
                    
            st.write("Step 3/5: Interactive Performance Heatmap (Test $R^2$)")
            r2_matrix = r2_matrix.astype(float).fillna(0.0)
            fig_heat = px.imshow(r2_matrix, text_auto=".2f", aspect="auto", 
                                 color_continuous_scale="RdBu_r", 
                                 title="Model vs Fingerprint Performance")
            st.plotly_chart(fig_heat, use_container_width=True)
            
            st.write("Step 4/5: Selecting Top Combinations for Stacking...")
            results_list.sort(key=lambda x: x[0], reverse=True)
            top_k = min(3, len(results_list))
            top_combinations = results_list[:top_k]
            
            for i, (score, m_name, f_name) in enumerate(top_combinations):
                st.success(f"🏅 Rank {i+1}: **{m_name}** with **{f_name}** (Base $R^2$: {score:.3f})")
                
            st.write(f"Step 5/5: Training Stacking Meta-Model with Top {top_k} Base Models...")
            
            meta_X_train = np.zeros((len(y_train), top_k))
            meta_X_test = np.zeros((len(y_test), top_k))
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for i, (score, m_name, f_name) in enumerate(top_combinations):
                base_model = MODELS_DICT[m_name]
                X_fp = fp_data_cache[f_name]
                X_tr, X_te = X_fp[idx_train], X_fp[idx_test]
                
                meta_X_train[:, i] = cross_val_predict(base_model, X_tr, y_train, cv=kf)
                base_model.fit(X_tr, y_train)
                meta_X_test[:, i] = base_model.predict(X_te)
                
            if X_env is not None:
                meta_X_train = np.hstack((meta_X_train, X_env[idx_train]))
                meta_X_test = np.hstack((meta_X_test, X_env[idx_test]))
                
            meta_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
            meta_model.fit(meta_X_train, y_train)
            
            y_pred_tr = meta_model.predict(meta_X_train)
            y_pred_te = meta_model.predict(meta_X_test)
            
            st.subheader("Final Stacking Ensemble Results")
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Train R²", f"{r2_score(y_train, y_pred_tr):.3f}")
            m_c2.metric("Test R²", f"{r2_score(y_test, y_pred_te):.3f}")
            m_c3.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_te)):.3f}")
            m_c4.metric("Test MAE", f"{mean_absolute_error(y_test, y_pred_te):.3f}")
            
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

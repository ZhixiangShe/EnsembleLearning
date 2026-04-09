import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
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
# 工具函数：绝对可靠的分子片段图像提取器 (防止图3画不出)
# ==========================================
def get_robust_bit_image_array(mol, bit, bit_info):
    try:
        img = Draw.DrawMorganBit(mol, int(bit), bit_info, useSVG=False)
        if hasattr(img, 'convert'):
            return np.array(img.convert('RGB'))
    except:
        pass
        
    try:
        atomId, radius = bit_info[bit][0]
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            h_atoms = [amap[atomId]] if atomId in amap else []
        else:
            submol = Chem.RWMol()
            submol.AddAtom(mol.GetAtomWithIdx(atomId))
            h_atoms = [0]
            
        d2d = rdMolDraw2D.MolDraw2DAGG(150, 150)
        opts = d2d.drawOptions()
        opts.clearBackground = True
        opts.setBackgroundColour((1, 1, 1, 1))
        d2d.DrawMolecule(submol, highlightAtoms=h_atoms)
        d2d.FinishDrawing()
        
        img_bytes = d2d.GetDrawingText()
        return np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    except Exception as e:
        return None

# ==========================================
# 1. 加载预训练模型 (带错误追踪的加强版)
# ==========================================
@st.cache_resource(show_spinner="Loading model files...")
def load_model_assets(dataset_name):
    file_map = {"O3": "model_assets_O3.pkl", "ZVI": "model_assets_ZVI.pkl"}
    path_in_folder = os.path.join("deploy_models", file_map[dataset_name])
    path_in_root = file_map[dataset_name]
    
    target_path = path_in_folder if os.path.exists(path_in_folder) else path_in_root if os.path.exists(path_in_root) else None
        
    # 如果找不到文件，直接在网页上爆红提示！
    if target_path is None:
        st.error(f"🚨 File Not Found: 服务器找不到文件 `{file_map[dataset_name]}`！请检查 GitHub 仓库。")
        return None
        
    try:
        with open(target_path, 'rb') as f: 
            return pickle.load(f)
    except Exception as e: 
        # 如果读取失败（如 LFS 问题），直接在网页打印报错信息！
        st.error(f"🚨 Model Load Error: 找到了文件 `{target_path}`，但是加载失败！")
        st.error(f"**错误详情:** {str(e)}")
        st.warning("💡 提示：如果是 `EOFError` 或 `unpickling error`，极有可能是因为模型超过 100MB，使用了 GitHub LFS，但 Streamlit 没有正确下载它（只下载了文本指针）。")
        return None

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
                cmap = cm.get_cmap('bwr')
                
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
                        # 换回强大无敌的图片生成器
                        img_arr = get_robust_bit_image_array(mol_shap, int(bit), bit_info)
                        
                        if img_arr is not None:
                            imagebox = OffsetImage(img_arr, zoom=0.6)
                            sign = 1 if top_vals[i] >= 0 else -1
                            offset = y_range * 0.15 
                            y_pos = top_vals[i] + (sign * offset)
                            
                            ab = AnnotationBbox(
                                imagebox, (i, y_pos), 
                                frameon=True, 
                                zorder=10,
                                bboxprops=dict(edgecolor='gray', boxstyle='round,pad=0.2', facecolor='white', alpha=0.9)
                            )
                            ax.add_artist(ab)
                            
                    ax.set_ylabel("Feature Importance (Contribution)")
                    ax.set_xlabel("ECFP Feature (Bit ID)")
                    ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
                    ax.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
                    
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
                if i % 10 == 0: pb.progress(

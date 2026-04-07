import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import streamlit.components.v1 as components  # 用于高清显示 SVG
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D       # 引入现代绘图引擎

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="QSAR/QSPR Intelligent Platform", layout="wide")

# ==========================================
# 1. 加载模型 (强力除错与多路径兼容)
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
        st.error(f"❌ Cannot find model file '{file_map[dataset_name]}'. Please check GitHub repository structure.")
        return None
        
    try:
        with open(target_path, 'rb') as f:
            assets = pickle.load(f)
        return assets
    except EOFError:
        st.error("❌ EOFError: Model file corrupted. This usually happens if the .pkl file is saved as a Git LFS pointer.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing required library: {e}. Please add it to requirements.txt.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__} - {str(e)}")
        return None

# ==========================================
# 2. 交互式预测核心函数 (支持 3D 构象生成)
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
        fp_calc = processor['fp_calculator']
        scaler = processor['scaler']
        imputer = processor['imputer']
        model = processor['model']
        
        try:
            fp_raw = fp_calc.transform([mol_3d])
            fp = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw)
            fp_processed = imputer.transform(scaler.transform(fp))
            pred = model.predict(fp_processed)[0]
            base_preds.append(pred)
        except Exception as e:
            return None, f"Error calculating {fp_name}: {str(e)}"
    
    env_features = [env_values_dict.get(name, 0.0) for name in assets['env_feature_names']]
    meta_input = np.array(base_preds + env_features).reshape(1, -1)
    
    meta_model = assets['meta_model']
    final_pred = meta_model.predict(meta_input)[0]
    
    return final_pred, mol_2d

# ==========================================
# 3. UI 布局
# ==========================================
st.title("QSAR/QSPR Intelligent Modeling & Prediction Platform")
st.markdown("A Visual Machine Learning Workflow for QSAR/QSPR Studies in Chemistry")

col_left, col_right = st.columns([1, 1])

# ------------- 左侧栏：项目配置 -------------
with col_left:
    st.header("Module 1: Project Configuration")
    dataset_choice = st.radio("Select Dataset Model", ["O3", "ZVI"], horizontal=True)
    st.button("Run Modeling Pipeline", disabled=True, help="Models are pre-trained for web deployment.")
    
    assets = load_model_assets(dataset_choice)
    
    st.header("Model Training Results")
    st.info("Static images (Heatmaps, Scatter plots) generated from local training can be displayed here.")

# ------------- 右侧栏：预测与可视化 -------------
with col_right:
    if assets is None:
        st.warning("Please resolve the model loading error to continue.")
    else:
        # --- 1. Interactive Prediction ---
        st.header("Interactive Analysis & Prediction")
        target_smiles = st.text_input("Enter Target Molecule SMILES", "CC(C)(C)C1=NN=C(S1)NC(=O)NC")
        
        env_inputs = {}
        if len(assets['env_feature_names']) > 0:
            st.write("Environmental Variables:")
            cols = st.columns(len(assets['env_feature_names']))
            for i, env_name in enumerate(assets['env_feature_names']):
                default_val = 7.0 if env_name.lower() == 'ph' else 25.0
                env_inputs[env_name] = cols[i].number_input(env_name, value=default_val)
                
        if st.button("Predict Kob Value", type="primary"):
            with st.spinner("Generating 3D conformer & Calculating Prediction..."):
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

        # --- 2. ECFP Atomic Contribution Visualization (彻底修复版) ---
        st.header("Molecular Interpretability (Atomic SHAP)")
        st.markdown("Visualizing atomic contributions using **ECFP (Morgan) Fingerprints** surrogate model.")
        
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
                            
                            # 步骤 A：单独计算每个原子的权重，不再调用旧版画图包
                            weights = SimilarityMaps.GetAtomicWeightsForModel(
                                mol_shap, 
                                lambda m, i: SimilarityMaps.GetMorganFingerprint(m, atomId=i, radius=2, nBits=2048), 
                                get_pred_for_shap
                            )
                            
                            # 步骤 B：使用 RDKit 现代的高清 SVG 引擎进行渲染
                            # 这样可以完全绕过 Matplotlib 的 MolToMPL 报错
                            d2d = rdMolDraw2D.MolDraw2DSVG(450, 450)
                            
                            SimilarityMaps.GetSimilarityMapFromWeights(
                                mol_shap, 
                                weights, 
                                colorMap='coolwarm', 
                                draw2d=d2d
                            )
                            d2d.FinishDrawing()
                            
                            # 获取生成的 SVG 文本
                            svg_text = d2d.GetDrawingText()
                            
                            # 在网页中渲染 SVG 图像
                            components.html(svg_text, width=450, height=450)
                            
                            st.markdown("""
                            **Interpretation Guide:**
                            * <span style='color:blue'>**Blue Areas**</span>: Atoms contributing **positively** to the prediction value.
                            * <span style='color:red'>**Red Areas**</span>: Atoms contributing **negatively** to the prediction value.
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
            else:
                st.error("Invalid SMILES.")

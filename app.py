import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
# --- 新增的除错代码 ---
st.warning("🕵️ Debugging Info (File System Check):")
st.write("1. Current directory path:", os.getcwd())
st.write("2. Files in current directory:", os.listdir('.'))

if os.path.exists('deploy_models'):
    st.write("3. Files inside 'deploy_models':", os.listdir('deploy_models'))
else:
    st.error("🚨 The folder 'deploy_models' DOES NOT EXIST in the current directory!")
# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="QSAR/QSPR Intelligent Platform", layout="wide")

# ==========================================
# 加载模型 (缓存以提高网页加载速度)
# ==========================================
@st.cache_resource(show_spinner="Loading model files...")
def load_model_assets(dataset_name):
    file_map = {
        "O3": "model_assets_O3.pkl",
        "ZVI": "model_assets_ZVI.pkl"
    }
    
    # 自动探测路径：优先找 deploy_models 文件夹，其次找根目录
    path_in_folder = os.path.join("deploy_models", file_map[dataset_name])
    path_in_root = file_map[dataset_name]
    
    target_path = None
    if os.path.exists(path_in_folder):
        target_path = path_in_folder
    elif os.path.exists(path_in_root):
        target_path = path_in_root
        
    if target_path is None:
        st.error(f"❌ 依然找不到文件！请检查大小写。")
        return None
        
    try:
        # st.info(f"✅ 成功找到文件所在路径: {target_path}，正在尝试读取...")
        with open(target_path, 'rb') as f:
            assets = pickle.load(f)
        return assets
    except EOFError:
        st.error(f"❌ 文件读取失败 (EOFError)！这通常是因为你的 .pkl 文件在 GitHub 上被存成了 Git LFS 指针。请确保上传的是真实的二进制文件。")
        return None
    except ModuleNotFoundError as e:
        st.error(f"❌ 缺少依赖库导致模型无法解包！请检查 requirements.txt: {e}")
        return None
    except Exception as e:
        st.error(f"❌ 读取模型时发生未知错误: {type(e).__name__} - {str(e)}")
        return None

# ==========================================
# 交互式预测核心函数 (Stacking 模型)
# ==========================================
def predict_smiles(smiles, env_values_dict, assets):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    
    base_preds = []
    top_fp_names = assets['top_fp_names']
    
    # 计算基础模型预测值
    for fp_name in top_fp_names:
        processor = assets['base_models_processors'][fp_name]
        fp_calc = processor['fp_calculator']
        scaler = processor['scaler']
        imputer = processor['imputer']
        model = processor['model']
        
        try:
            fp_raw = fp_calc.transform([mol])
            fp = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw)
            fp_processed = imputer.transform(scaler.transform(fp))
            pred = model.predict(fp_processed)[0]
            base_preds.append(pred)
        except Exception as e:
            return None, f"Error calculating {fp_name}: {str(e)}"
    
    # 合并环境特征
    env_features = [env_values_dict.get(name, 0.0) for name in assets['env_feature_names']]
    meta_input = np.array(base_preds + env_features).reshape(1, -1)
    
    # Meta-model 最终预测
    meta_model = assets['meta_model']
    final_pred = meta_model.predict(meta_input)[0]
    
    return final_pred, mol

# ==========================================
# UI 布局
# ==========================================
st.title("QSAR/QSPR Intelligent Modeling & Prediction Platform")
st.markdown("A Visual Machine Learning Workflow for QSAR/QSPR Studies in Chemistry")

col_left, col_right = st.columns([1, 1])

# ------------- 左侧栏：项目配置 -------------
with col_left:
    st.header("Module 1: Project Configuration")
    
    # 【修改点2】将选项卡更新为 O3 和 ZVI
    dataset_choice = st.radio("Select Dataset Model", ["O3", "ZVI"], horizontal=True)
    st.button("Run Modeling Pipeline", disabled=True, help="Models are pre-trained for web deployment.")
    
 # 加载选中的模型
    assets = load_model_assets(dataset_choice)
    
    st.header("Model Training Results")
    st.info("Here you can place static images (Heatmaps, Scatter plots) generated from your local training phase.")
    # 如果你有训练结果图片，可以取消下面这行的注释并修改路径
    # st.image("results/performance_plot.png")

# ------------- 右侧栏：预测与可视化 -------------
with col_right:
    if assets is None:
        st.error(f"Model file 'model_assets_{dataset_choice}.pkl' not found. Please upload it to 'deploy_models' folder.")
    else:
        # --- 1. Interactive Prediction ---
        st.header("Interactive Analysis & Prediction")
        target_smiles = st.text_input("Enter Target Molecule SMILES", "CC(C)(C)C1=NN=C(S1)NC(=O)NC")
        
        # 动态生成环境参数输入框
        env_inputs = {}
        if len(assets['env_feature_names']) > 0:
            st.write("Environmental Variables:")
            cols = st.columns(len(assets['env_feature_names']))
            for i, env_name in enumerate(assets['env_feature_names']):
                # 默认值: pH给7.0, 其他给25.0
                env_inputs[env_name] = cols[i].number_input(env_name, value=7.0 if env_name.lower() == 'ph' else 25.0)
                
        if st.button("Predict Kob Value", type="primary"):
            with st.spinner("Calculating Prediction..."):
                pred_value, mol = predict_smiles(target_smiles, env_inputs, assets)
                if pred_value is not None:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption="2D Molecular Structure")
                    with c2:
                        st.success("Prediction Complete!")
                        st.metric(label="Predicted Kob Value", value=f"{pred_value:.4f}")
                else:
                    st.error(mol)

        st.divider()

        # --- 2. ECFP Atomic Contribution Visualization (SHAP 近似) ---
        st.header("Molecular Interpretability (Atomic SHAP)")
        st.markdown("Visualizing atomic contributions using **ECFP (Morgan) Fingerprints** surrogate model.")
        
        shap_smiles = st.text_input("Enter SMILES for Interpretability", target_smiles, key="shap_input")
        
        if st.button("Generate Atomic Contribution Plot"):
            mol_shap = Chem.MolFromSmiles(shap_smiles)
            if mol_shap:
                with st.spinner("Calculating ECFP Atomic Contributions..."):
                    try:
                        # 获取在本地训练的 ECFP 代理模型
                        if 'ecfp_surrogate' not in assets:
                            st.error("ECFP Surrogate model not found in the .pkl file. Please ensure you ran the surrogate training step locally.")
                        else:
                            ecfp_model = assets['ecfp_surrogate']
                            
                            # 定义 RDKit SimilarityMaps 需要的回调预测函数
                            def get_pred_for_shap(fp_vect):
                                fp_array = np.array(fp_vect).reshape(1, -1)
                                return ecfp_model.predict(fp_array)[0]
                            
                            # 生成原子热力图
                            fig, maxweight = SimilarityMaps.GetSimilarityMapForModel(
                                mol_shap, 
                                lambda m, i: SimilarityMaps.GetMorganFingerprint(m, atomId=i, radius=2, nBits=2048), 
                                get_pred_for_shap, 
                                colorMap='coolwarm' # 蓝红配色
                            )
                            
                            st.pyplot(fig)
                            st.markdown("""
                            **Interpretation Guide:**
                            * <span style='color:blue'>**Blue Areas**</span>: Atoms contributing **positively** to the prediction value.
                            * <span style='color:red'>**Red Areas**</span>: Atoms contributing **negatively** to the prediction value.
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
            else:
                st.error("Invalid SMILES.")

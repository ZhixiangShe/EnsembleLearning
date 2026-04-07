import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
from skfp.preprocessing import ConformerGenerator

# --- 页面基础配置 ---
st.set_page_config(page_title="QSAR Prediction Platform", layout="wide")
st.title("🧪 QSAR/QSPR Prediction Platform")


# ==========================================
# 1. 缓存加载模型 (核心优化)
# ==========================================
# 使用 @st.cache_resource 装饰器，确保每次刷新网页时不用重新加载几百MB的模型
@st.cache_resource
def load_model():
    # 请确保将训练好的 .pkl 文件放在与 app.py 同级的目录下
    # 替换为你实际生成的文件名
    model_path = 'qsar_production_model_O3 all new.pkl'
    return joblib.load(model_path)


try:
    with st.spinner("Loading AI Models..."):
        model_data = load_model()
        top_fp_names = model_data['top_fp_names']
        base_components = model_data['base_components']
        meta_model = model_data['meta_model']
        env_feature_names = model_data['env_feature_names']
        dataset_name = model_data['dataset_name']
    st.success(f"Model trained on '{dataset_name}' loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 2. 构建用户界面 (UI)
# ==========================================
st.markdown("---")
st.subheader("Input Parameters")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**1. Molecule Structure**")
    smiles_input = st.text_input("Enter SMILES string:", "CC(C)(C)C1=NN=C(S1)NC(=O)NC")

    # 实时显示分子结构图
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption="2D Structure", width=250)
    else:
        st.warning("Please enter a valid SMILES string.")

with col2:
    st.markdown("**2. Environmental Conditions**")
    env_inputs = {}
    # 动态生成环境特征的输入框（根据你打包时的 env_feature_names）
    if len(env_feature_names) > 0:
        for feature in env_feature_names:
            # 你可以设置默认值，这里统一默认为 0.0
            env_inputs[feature] = st.number_input(f"{feature}:", value=0.0, format="%.4f")
    else:
        st.info("This model does not require environmental features.")

# ==========================================
# 3. 推理预测逻辑 (Inference Logic)
# ==========================================
st.markdown("---")
if st.button("🚀 Predict Kob Value", use_container_width=True, type="primary"):
    if not mol:
        st.error("Cannot predict: Invalid SMILES.")
    else:
        with st.spinner("Calculating molecular fingerprints and generating predictions..."):
            try:
                # 步骤 A: 准备分子输入 (处理 3D 构象)
                mols_2d = [mol]
                mols_3d = None

                # 检查是否需要 3D 构象
                needs_3d = any(comp['needs_conformers'] for comp in base_components.values())
                if needs_3d:
                    conf_gen = ConformerGenerator(num_conformers=1, random_state=42)
                    mols_3d = conf_gen.transform(mols_2d)
                    if mols_3d[0] is None or mols_3d[0].GetNumConformers() == 0:
                        st.error("Failed to generate 3D conformer for this molecule.")
                        st.stop()

                # 步骤 B: 基础模型预测 (收集 Meta-Features)
                base_predictions = []
                for fp_name in top_fp_names:
                    comp = base_components[fp_name]
                    fp_calc = comp['fp_calculator']
                    scaler = comp['scaler']
                    imputer = comp['imputer']
                    base_model = comp['model']

                    # 1. 计算指纹
                    target_mol = mols_3d if comp['needs_conformers'] else mols_2d
                    fp_raw = fp_calc.transform(target_mol)
                    fp_array = np.asarray(fp_raw.toarray() if hasattr(fp_raw, "toarray") else fp_raw)

                    # 2. 缩放与插值 (必须和训练时完全一致)
                    fp_scaled = scaler.transform(fp_array)
                    fp_imputed = imputer.transform(fp_scaled)

                    # 3. 基础模型预测
                    pred = base_model.predict(fp_imputed)[0]
                    base_predictions.append(pred)

                # 步骤 C: 组装最终特征矩阵进行 Meta-Model 预测
                # Meta特征格式: [base_pred_1, base_pred_2, ..., env_feature_1, env_feature_2...]
                meta_features = np.array(base_predictions)

                if len(env_feature_names) > 0:
                    env_values = np.array([env_inputs[feat] for feat in env_feature_names])
                    final_input = np.concatenate((meta_features, env_values)).reshape(1, -1)
                else:
                    final_input = meta_features.reshape(1, -1)

                # 步骤 D: 最终预测
                final_prediction = meta_model.predict(final_input)[0]

                # --- 显示结果 ---
                st.success("Prediction Complete!")

                # 大字展示最终结果
                st.metric(label="Predicted Target Value (Kob)", value=f"{final_prediction:.4f}")

                # 可选：展示底层模型的预测细节（增加解释性）
                with st.expander("View Base Models Detail"):
                    detail_df = pd.DataFrame({
                        "Base Fingerprint Model": top_fp_names,
                        "Prediction": [f"{p:.4f}" for p in base_predictions]
                    })
                    st.table(detail_df)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
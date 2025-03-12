###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('LGBM_pos.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature names
feature_names = ['PASP','Pre_hemoglobin','LAD', 'ACC_duration', 'Intra_urine_output','MAP_65_time','Post_CPB_MAP_AUT_65','Post_CPB_CVP_12_time']

## Streamlit user interface
st.title("HT-AKI Predictor")

# PASP: numerical input
PASP = st.number_input("PASP (mmHg):", min_value=10, max_value=100, value=30)
# Pre_hemoglobin: numerical input
Pre_hemoglobin = st.number_input("Pre hemoglobin (g/L):", min_value=50, max_value=200, value=120)
# LAD: numerical input
LAD = st.number_input("LAD (mm):", min_value=40, max_value=100, value=50)
# ACC_duration：numerical input
ACC_duration = st.number_input("ACC duration (min):", min_value=30, max_value=200, value=90)
# Intra_urine_output：numerical input
Intra_urine_output =  st.number_input("Intra urine output (ml/kg/h):", min_value=0.00, max_value=10.00, value=0.00)
# MAP_65_time: numerical input
MAP_65_time = st.number_input("MAP < 65 time (min):", min_value=0, max_value=400, value=0)
# Post_CPB_MAP_AUT_65: numerical input
Post_CPB_MAP_AUT_65 = st.number_input("Post_CPB_MAP_AUT_65 (min·mmHg):", min_value=0, max_value=4000, value=0)
# Post_CPB_CVP_12_time: numerical input
Post_CPB_CVP_12_time = st.number_input("Post_CPB_CVP > 12 time (min):", min_value=0, max_value=200, value=0)

# Process inputs and make predictions
feature_values = [PASP,Pre_hemoglobin,LAD,ACC_duration,Intra_urine_output,MAP_65_time,Post_CPB_MAP_AUT_65,Post_CPB_CVP_12_time]
features = np.array([feature_values])

# 关键修改：使用 pandas DataFrame 来确保列名
features_df = pd.DataFrame(features, columns=feature_names)
standardized_features_1 = scaler.transform(features_df)

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
standardized_features = pd.DataFrame(standardized_features_1, columns=feature_names)

if st.button("Predict"):    
    # 标准化特征
    # standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
     if predicted_class == 1:        
         advice = (            
                f"According to our model prediction, you have a high risk of AKI after heart transplant surgery. "            
                f"The model predicts that your probability of having AKI is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model prediction, you have a low risk of heart disease. "            
                f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame(standardized_features, columns=feature_names))
    
    # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    # 检查 shap_values 的格式
    print("SHAP values type:", type(shap_values))
    print("SHAP values shape:", [np.shape(v) for v in shap_values] if isinstance(shap_values, list) else np.shape(shap_values))
    
    # 根据 shap_values 的格式调整代码
    if isinstance(shap_values, list):  # 如果是列表（二分类模型）
        if predicted_class == 1:
            shap.force_plot(explainer_shap.expected_value[1], shap_values[1], original_feature_values, matplotlib=True)
        else:
            shap.force_plot(explainer_shap.expected_value[0], shap_values[0], original_feature_values, matplotlib=True)
    else:  # 如果是 NumPy 数组（回归模型或单输出模型）
        shap.force_plot(explainer_shap.expected_value, shap_values, original_feature_values, matplotlib=True)
    
    # 保存并显示图像
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

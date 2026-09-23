import joblib
import pandas as pd
import os

# 預期特徵順序 (必須與訓練時完全一致)
FEATURES = [
    'energy_100g', 
    'proteins_100g', 
    'fat_100g', 
    'carbohydrates_100g', 
    'sugars_100g', 
    'salt_100g'
]

# Label Mapping (A=0, B=1, C=2, D=3, E=4)
LABEL_MAP = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

def load_model():
    """載入既有的 Random Forest 模型"""
    # 假設從 app.py 啟動，相對路徑指向 04_Model
    model_path = os.path.join('..', '04_Model', 'models', 'random_forest_best.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model = joblib.load(model_path)
    return model

def predict_nutrition_grade(model, user_input: dict) -> str:
    """
    接收字典格式的輸入特徵，進行格式轉換後送入模型預測。
    回傳 'A', 'B', 'C', 'D', 或 'E'。
    """
    # 建立 DataFrame 並確保順序與欄位正確
    input_df = pd.DataFrame([user_input])[FEATURES]
    
    # 執行預測
    prediction_int = model.predict(input_df)[0]
    
    # 將整數 mapping 回對應的英文字母
    prediction_label = LABEL_MAP.get(prediction_int, 'Unknown')
    return prediction_label

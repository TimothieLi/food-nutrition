import streamlit as st
from PIL import Image
import os
import sys

# 將 utils 目錄加入環境路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ocr_engine import extract_nutrition_info, is_nutrition_info_found
from utils.model_runner import load_model, predict_nutrition_grade

# 設定頁面配置
st.set_page_config(
    page_title="AI 食品營養標示分析器",
    page_icon="🍏",
    layout="centered"
)

# 載入模型 (使用快取以避免重複讀取)
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"模型載入失敗：{e}")
        return None

model = get_model()

# 初始化 Session State
if 'ocr_data' not in st.session_state:
    st.session_state.ocr_data = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False

st.title("🍏 AI 食品營養標示分析器")
st.subheader("拍一拍，AI 幫你看懂食品營養標示")

st.markdown("---")

# 上傳區塊
st.write("### Step 1: 拍攝或上傳食品營養標示")
st.write("請對準食品包裝後方的營養成分表進行拍攝。")

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    camera_img = st.camera_input("使用相機拍照")
with upload_col2:
    uploaded_file = st.file_uploader("或上傳圖片檔", type=['png', 'jpg', 'jpeg'])

# 處理輸入影像
input_image = None
if camera_img is not None:
    input_image = Image.open(camera_img)
elif uploaded_file is not None:
    input_image = Image.open(uploaded_file)

if input_image is not None and not st.session_state.image_uploaded:
    st.session_state.image_uploaded = True
    st.info("🔄 正在進行 OCR 文字辨識...")
    
    # 進行 OCR
    ocr_results = extract_nutrition_info(input_image)
    
    if not is_nutrition_info_found(ocr_results):
        st.warning("⚠️ 找不到足夠的營養資訊。請拍攝食品包裝上的營養標示，或改用手動輸入。")
        st.session_state.ocr_data = {k: 0.0 for k in ocr_results.keys()} # 提供全空預設值
    else:
        # 檢查是否有缺漏
        missing_fields = [k for k, v in ocr_results.items() if v is None]
        if missing_fields:
            st.warning("⚠️ 營養資訊辨識不完整。請手動補充缺漏的欄位。")
        else:
            st.success("✅ 辨識完成！")
        
        # 補齊 None 為 0.0，方便顯示於表單
        for k in ocr_results:
            if ocr_results[k] is None:
                ocr_results[k] = 0.0
        
        st.session_state.ocr_data = ocr_results

elif input_image is None:
    st.session_state.image_uploaded = False
    st.session_state.ocr_data = None


# 手動修改與確認區塊
st.markdown("---")
st.write("### Step 2: 確認與修改營養資訊 (每 100g)")

# 無論有無照片，都允許使用者手動輸入
if st.session_state.ocr_data is None:
    # 預設全為 0 的表單
    current_data = {
        'energy_100g': 0.0,
        'proteins_100g': 0.0,
        'fat_100g': 0.0,
        'carbohydrates_100g': 0.0,
        'sugars_100g': 0.0,
        'salt_100g': 0.0
    }
else:
    current_data = st.session_state.ocr_data

st.info("AI 辨識到以下營養資訊，請確認是否正確。您可以隨時手動修改。")

with st.form("nutrition_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        energy = st.number_input("Energy (kcal)", value=float(current_data['energy_100g']), min_value=0.0, step=1.0)
        protein = st.number_input("Protein (g)", value=float(current_data['proteins_100g']), min_value=0.0, step=0.1)
        fat = st.number_input("Fat (g)", value=float(current_data['fat_100g']), min_value=0.0, step=0.1)
    
    with col2:
        carbs = st.number_input("Carbohydrates (g)", value=float(current_data['carbohydrates_100g']), min_value=0.0, step=0.1)
        sugars = st.number_input("Sugars (g)", value=float(current_data['sugars_100g']), min_value=0.0, step=0.1)
        salt = st.number_input("Salt (g)", value=float(current_data['salt_100g']), min_value=0.0, step=0.01)

    submitted = st.form_submit_button("✅ 確認並開始 AI 分析", type="primary", use_container_width=True)

# 執行預測
if submitted:
    st.markdown("---")
    st.write("### Step 3: AI 分析結果")
    
    if model is None:
        st.error("模型載入失敗，無法進行分析。")
    else:
        user_input = {
            'energy_100g': energy,
            'proteins_100g': protein,
            'fat_100g': fat,
            'carbohydrates_100g': carbs,
            'sugars_100g': sugars,
            'salt_100g': salt
        }
        
        # 進行預測
        grade = predict_nutrition_grade(model, user_input)
        
        # 顏色設定
        color_map = {'A': '#008b45', 'B': '#85bb2f', 'C': '#fecb02', 'D': '#ee8100', 'E': '#e63e11', 'Unknown': 'gray'}
        grade_color = color_map.get(grade, 'gray')
        
        # 顯示大大的結果
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 2rem;'>
            <h2 style='color: #31333F;'>Nutrition Grade</h2>
            <h1 style='font-size: 5rem; color: {grade_color}; margin: 0;'>{grade}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("#### 本次分析使用的營養資訊：")
        st.json({
            "Energy (kcal/100g)": energy,
            "Protein (g/100g)": protein,
            "Fat (g/100g)": fat,
            "Carbohydrates (g/100g)": carbs,
            "Sugars (g/100g)": sugars,
            "Salt (g/100g)": salt
        })
        
        st.write("#### 模型資訊：")
        st.code("Model: Random Forest\nTest Accuracy: 76.12%")
        
        st.markdown("---")
        st.markdown("#### 📖 關於 Nutrition Grade")
        st.caption("A / B / C / D / E 是 Open Food Facts 資料中的 Nutrition Grade。")
        st.caption("Demo 中的 Machine Learning 模型是學習食品營養特徵與資料集中 Nutrition Grade 之間的關係，進而對未知食品進行 A～E Classification 預測。")
        
        st.warning("⚠️ **資訊素養提醒**\n\n1. 此結果是 Machine Learning 模型根據輸入特徵所做的分類預測，**不代表完整的健康評估**，亦非醫療建議。\n2. **AI 預測並非絕對正確**。\n3. OCR 辨識結果可能出現錯誤，請隨時留意數值是否合理。")

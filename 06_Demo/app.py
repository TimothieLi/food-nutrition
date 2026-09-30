import streamlit as st
from PIL import Image
import os
import sys

# 將 utils 目錄加入環境路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()
from utils.llm_vision import extract_nutrition_info
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
    
    with st.spinner("🔄 正在透過 Vision LLM 辨識營養標示..."):
        # 進行 OCR
        ocr_results, error_msg = extract_nutrition_info(input_image)
    
    # 預設全空數值
    default_empty = {k: 0.0 for k in ['energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'salt_100g']}
    
    if error_msg:
        st.error(error_msg)
        st.session_state.ocr_data = default_empty
    elif ocr_results.get("status") == "not_found":
        st.warning("⚠️ 找不到足夠的營養資訊。\n請確認照片中包含完整的食品營養標示。")
        st.session_state.ocr_data = default_empty
    elif ocr_results.get("status") == "uncertain":
        st.warning("⚠️ 無法辨識這張圖片中的完整營養標示，請重新拍攝或手動確認。")
        st.session_state.ocr_data = default_empty
    else:
        # success 狀態
        missing_fields = [k for k in default_empty.keys() if ocr_results.get(k) is None]
        if missing_fields:
            st.warning("⚠️ 部分營養資訊無法可靠辨識，請重新拍攝或手動確認。")
        else:
            st.success("✅ 營養資訊辨識完成！")
        
        # 補齊 None 為 0.0，方便顯示於表單
        for k in default_empty.keys():
            if ocr_results.get(k) is None:
                ocr_results[k] = 0.0
        
        st.session_state.ocr_data = ocr_results

elif input_image is None:
    st.session_state.image_uploaded = False
    st.session_state.ocr_data = None


# 手動修改與確認區塊
st.markdown("---")
st.write("### Step 2：確認營養資訊（每 100 公克）")

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

st.info("AI 已辨識以下營養資訊，請確認是否正確；如有需要，可以直接修改。")

with st.form("nutrition_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        energy_kcal = st.number_input("熱量 (kcal)", value=float(current_data['energy_100g']), min_value=0.0, step=1.0)
        protein = st.number_input("蛋白質 (g)", value=float(current_data['proteins_100g']), min_value=0.0, step=0.1)
        fat = st.number_input("脂肪 (g)", value=float(current_data['fat_100g']), min_value=0.0, step=0.1)
    
    with col2:
        carbs = st.number_input("碳水化合物 (g)", value=float(current_data['carbohydrates_100g']), min_value=0.0, step=0.1)
        sugars = st.number_input("糖 (g)", value=float(current_data['sugars_100g']), min_value=0.0, step=0.1)
        salt = st.number_input("食鹽 (g)", value=float(current_data['salt_100g']), min_value=0.0, step=0.01)

    submitted = st.form_submit_button("確認", type="primary", use_container_width=True)

# 執行預測
if submitted:
    st.markdown("---")
    
    if model is None:
        st.error("模型載入失敗，無法進行分析。")
    else:
        user_input = {
            'energy_100g': energy_kcal * 4.184,  # 將 kcal 轉換回 kJ 送進模型
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
        
        # ━━━━━━━━━━━━━━━━━━━━
        # 【1. 🥗 AI 預測營養分級】
        # ━━━━━━━━━━━━━━━━━━━━
        st.write("### 🥗 AI 預測營養分級")
        
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 2rem;'>
            <h2 style='color: #31333F;'>Nutrition Grade</h2>
            <h1 style='font-size: 5rem; color: {grade_color}; margin: 0;'>{grade}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        def get_badge_html(target_grade, current_grade, color):
            bg = color if target_grade == current_grade else '#e0e0e0'
            return f"<div style='background-color: {bg}; color: white; padding: 5px; border-radius: 50%; font-weight: bold; font-size: 1.5rem; display: flex; align-items: center; justify-content: center; width: 48px; height: 48px;'>{target_grade}</div>"

        html_badges = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 400px; margin: 1.5rem auto 1rem auto;">
            {get_badge_html('A', grade, '#008b45')}
            <div style="color: #ccc; font-size: 1.5rem;">➔</div>
            {get_badge_html('B', grade, '#85bb2f')}
            <div style="color: #ccc; font-size: 1.5rem;">➔</div>
            {get_badge_html('C', grade, '#fecb02')}
            <div style="color: #ccc; font-size: 1.5rem;">➔</div>
            {get_badge_html('D', grade, '#ee8100')}
            <div style="color: #ccc; font-size: 1.5rem;">➔</div>
            {get_badge_html('E', grade, '#e63e11')}
        </div>
        <div style="display: flex; justify-content: space-between; max-width: 400px; margin: 0.8rem auto 2rem auto; font-size: 1rem; color: gray; font-weight: 500;'>
            <span>較佳</span>
            <span>較需留意</span>
        </div>
        """
        st.markdown(html_badges, unsafe_allow_html=True)
        
        grade_desc = {
            'A': '較佳',
            'B': '良好',
            'C': '中等',
            'D': '需留意',
            'E': '較需留意'
        }.get(grade, '未知')
        
        st.write(f"此食品在本 Demo 的營養分類模型中，被判定為 **{grade} 級**，屬於**{grade_desc}**的分類。")
        
        # ━━━━━━━━━━━━━━━━━━━━
        # 【2. 💡 營養觀察】
        # ━━━━━━━━━━━━━━━━━━━━
        st.write("### 💡 營養觀察")
        with st.spinner("🔄 正在產生營養觀察..."):
            from utils.llm_vision import generate_nutrition_advice
            nutrition_for_prompt = {
                'energy_100g': energy_kcal,
                'proteins_100g': protein,
                'fat_100g': fat,
                'carbohydrates_100g': carbs,
                'sugars_100g': sugars,
                'salt_100g': salt
            }
            advice = generate_nutrition_advice(nutrition_for_prompt, grade)
            st.write(advice)
            
        # ━━━━━━━━━━━━━━━━━━━━
        # 【3. ⚠️ 使用提醒】
        # ━━━━━━━━━━━━━━━━━━━━
        st.markdown("---")
        st.caption("⚠️ **使用提醒**：AI 預測僅供參考，並非健康診斷；請確認營養標示與食用份量後再做判斷。")

import streamlit as st
from PIL import Image
import os
import sys

# 將 utils 目錄加入環境路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()
from utils.llm_vision import extract_nutrition_info, generate_nutrition_advice
from utils.model_runner import load_model, predict_nutrition_grade
from utils.validator import validate_nutrition_data, BASIS_DISPLAY_NAMES

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
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'last_validation' not in st.session_state:
    st.session_state.last_validation = None

st.title("🍏 AI 食品營養標示分析器")
st.subheader("拍一拍，AI 幫你看懂食品營養標示")

st.markdown("---")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【Step 1: 拍攝或上傳食品營養標示】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    # 圖片變更時清空舊的預測與驗證狀態
    st.session_state.prediction_result = None
    st.session_state.last_validation = None
    
    with st.spinner("🔄 正在透過 Vision LLM 辨識營養標示..."):
        # 進行 OCR
        ocr_results, error_msg = extract_nutrition_info(input_image)
    
    # 預設全空（None）結構，嚴禁將未取得之數值預設為 0.0
    default_empty = {
        'energy_100g': None,
        'proteins_100g': None,
        'fat_100g': None,
        'carbohydrates_100g': None,
        'sugars_100g': None,
        'salt_100g': None,
        'basis': 'unknown'
    }
    
    if error_msg:
        st.error(error_msg)
        st.session_state.ocr_data = default_empty
    elif ocr_results.get("status") == "not_found":
        st.warning("⚠️ 找不到足夠的營養資訊。\n請確認照片中包含完整的食品營養標示。")
        st.session_state.ocr_data = default_empty
    elif ocr_results.get("status") == "uncertain":
        st.warning("⚠️ 無法可靠辨識這張圖片中的完整營養標示，請重新拍攝或手動輸入確認。")
        st.session_state.ocr_data = default_empty
    else:
        # success 狀態
        missing_fields = [k for k in ['energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'salt_100g'] if ocr_results.get(k) is None]
        if missing_fields:
            st.warning("⚠️ 部分營養資訊無法由 OCR 可靠辨識，請在下方手動補齊。")
        else:
            st.success("✅ 營養資訊辨識完成！")
        
        # 檢查 OCR 辨識出的標示基準
        basis_detected = ocr_results.get("basis", "per_100g")
        if basis_detected in ["per_serving", "per_package", "unknown"]:
            st.warning(f"⚠️ 注意：OCR 辨識出之原始標示基準可能為『{basis_detected}』非每 100g。本系統分析以『每 100 公克』為準，請務必手動換算為每 100g 數值後確認。")
        
        # 保留 None，絕不自動覆寫為 0.0
        st.session_state.ocr_data = ocr_results

elif input_image is None and st.session_state.image_uploaded:
    # 使用者移除了圖片
    st.session_state.image_uploaded = False
    st.session_state.ocr_data = None
    st.session_state.prediction_result = None
    st.session_state.last_validation = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【Step 2: 手動修改與確認營養資訊】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.write("### Step 2：確認營養資訊（每 100 公克）")

# 無論有無照片，都允許使用者手動輸入或檢查
if st.session_state.ocr_data is None:
    current_data = {
        'energy_100g': None,
        'proteins_100g': None,
        'fat_100g': None,
        'carbohydrates_100g': None,
        'sugars_100g': None,
        'salt_100g': None,
        'basis': 'per_100g'
    }
else:
    current_data = st.session_state.ocr_data

st.info("請確認以下營養標示資訊是否正確完整；如有遺漏或誤差，可直接修正。")

# 標示基準選項
basis_options = [
    "每 100 公克 (per 100g)",
    "每 100 毫升 (per 100ml)",
    "每份 (per serving)",
    "每包 / 整包 (per package)",
    "無法確認基準 (unknown)"
]
basis_code_map = {
    "每 100 公克 (per 100g)": "per_100g",
    "每 100 毫升 (per 100ml)": "per_100ml",
    "每份 (per serving)": "per_serving",
    "每包 / 整包 (per package)": "per_package",
    "無法確認基準 (unknown)": "unknown"
}

curr_basis_code = current_data.get('basis', 'per_100g')
default_basis_idx = 0
for idx, label in enumerate(basis_options):
    if basis_code_map[label] == curr_basis_code:
        default_basis_idx = idx
        break

selected_basis_label = st.selectbox(
    "原始標示之計算基準：",
    options=basis_options,
    index=default_basis_idx,
    help="本系統機器學習模型以『每 100 公克 (per 100g)』為標準特徵。若標示為每份或整包，請先手動依份量換算為每 100g。"
)
selected_basis = basis_code_map[selected_basis_label]

def to_float_or_none(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

with st.form("nutrition_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        energy_kcal = st.number_input(
            "熱量 (kcal)", 
            value=to_float_or_none(current_data.get('energy_100g')), 
            step=1.0,
            placeholder="請輸入熱量 (kcal)"
        )
        protein = st.number_input(
            "蛋白質 (g)", 
            value=to_float_or_none(current_data.get('proteins_100g')), 
            step=0.1,
            placeholder="請輸入蛋白質 (g)"
        )
        fat = st.number_input(
            "脂肪 (g)", 
            value=to_float_or_none(current_data.get('fat_100g')), 
            step=0.1,
            placeholder="請輸入脂肪 (g)"
        )
    
    with col2:
        carbs = st.number_input(
            "碳水化合物 (g)", 
            value=to_float_or_none(current_data.get('carbohydrates_100g')), 
            step=0.1,
            placeholder="請輸入碳水化合物 (g)"
        )
        sugars = st.number_input(
            "糖 (g)", 
            value=to_float_or_none(current_data.get('sugars_100g')), 
            step=0.1,
            placeholder="請輸入糖 (g)"
        )
        salt = st.number_input(
            "食鹽 (g)", 
            value=to_float_or_none(current_data.get('salt_100g')), 
            step=0.01,
            placeholder="請輸入食鹽 (g)"
        )

    # 全零數值確認勾選框
    confirm_all_zero = st.checkbox(
        "我確認此食品之六項營養數值確實皆為 0（例如：包裝純水、無糖無熱量飲品）",
        value=False,
        help="若六項數值皆為 0 且未勾選確認，系統將判定為待確認之異常情境並不予預測。"
    )

    submitted = st.form_submit_button("確認並進行分析", type="primary", use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【表單送出與驗證流程】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if submitted:
    # 重要防護：每次送出表單時，立即清除先前殘留的預測結果與營養分析文字
    st.session_state.prediction_result = None
    st.session_state.last_validation = None

    raw_inputs = {
        'energy_kcal': energy_kcal,
        'proteins_100g': protein,
        'fat_100g': fat,
        'carbohydrates_100g': carbs,
        'sugars_100g': sugars,
        'salt_100g': salt
    }

    # 執行嚴格的輸入資料驗證
    val_result = validate_nutrition_data(
        data=raw_inputs,
        basis=selected_basis,
        confirmed_all_zero=confirm_all_zero
    )
    st.session_state.last_validation = val_result

    if not val_result['is_valid']:
        # 驗證未通過或需額外確認：顯示錯誤與警告，阻止進入模型預測與營養觀察
        if val_result['errors']:
            st.error("❌ 輸入資料驗證未通過，無法進行預測：")
            for err in val_result['errors']:
                st.markdown(f"- {err}")
        if val_result['warnings']:
            for warn in val_result['warnings']:
                st.warning(f"⚠️ {warn}")
    else:
        # 驗證通過：進入分析流程
        if model is None:
            st.error("模型載入失敗，無法進行分析。")
        else:
            cleaned_input = val_result['cleaned_data']
            is_all_zero = val_result['info'].get('all_zero', False)
            
            # 使用 spinner 同時包覆模型預測與營養觀察，在此期間絕不提前單獨渲染分級畫面
            with st.spinner("🔄 正在進行營養評估與生成營養觀察..."):
                # 1. 執行 Random Forest 模型預測
                grade = predict_nutrition_grade(model, cleaned_input)
                
                # 2. 產生營養觀察（若為全零則給予四段式客觀說明，否則呼叫 LLM 服務）
                if is_all_zero:
                    advice_text = (
                        "**🥗 營養特徵摘要**\n"
                        "此食品標示之熱量、蛋白質、脂肪、碳水化合物、糖及食鹽含量皆為 0（常見於包裝飲用水或無糖無熱量純茶），屬於不含三大營養素與鈉負擔的零熱量純水分飲品。\n\n"
                        "**⚠️ 值得注意的營養項目**\n"
                        "• **熱量與必需營養素**：各項數值皆標示為 0，代表此食品無法提供人體日常活動所需之熱量、蛋白質或必需脂肪酸。\n\n"
                        "**💡 飲食建議**\n"
                        "1. **優良水分補給**：適合作為日常解渴與水分補充來源，無額外精緻糖分或熱量負擔。\n"
                        "2. **配合多元正餐**：本品不可取代正餐或作為單一營養來源，日常仍需搭配多樣化膳食以攝取必要營養。\n\n"
                        "**📌 整體評估**\n"
                        "根據模型預測結果，本食品被評為 A 級，反映其不含額外負擔之成分。但健康飲食需考量整體膳食搭配與個人活動量，不可僅依賴分級判定單一食品之營養價值。"
                    )
                    advice_error = None
                else:
                    nutrition_for_prompt = {
                        'energy_100g': raw_inputs['energy_kcal'],
                        'proteins_100g': raw_inputs['proteins_100g'],
                        'fat_100g': raw_inputs['fat_100g'],
                        'carbohydrates_100g': raw_inputs['carbohydrates_100g'],
                        'sugars_100g': raw_inputs['sugars_100g'],
                        'salt_100g': raw_inputs['salt_100g']
                    }
                    advice_text, advice_error = generate_nutrition_advice(
                        nutrition_for_prompt, grade, basis=selected_basis
                    )
            
            # 3. 將分析結果完整記錄至 Session State（統一於下方渲染區塊一次顯示）
            st.session_state.prediction_result = {
                'grade': grade,
                'inputs': cleaned_input,
                'raw_inputs': raw_inputs,
                'is_all_zero': is_all_zero,
                'advice': advice_text,
                'advice_error': advice_error
            }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【Step 3: 統一顯示分析結果（分級與營養觀察）】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 唯有驗證通過且預測與觀察流程執行完畢時，才一次完整顯示整合區塊
if st.session_state.prediction_result is not None:
    res = st.session_state.prediction_result
    grade = res['grade']
    is_all_zero = res['is_all_zero']
    raw_inputs = res['raw_inputs']
    advice = res.get('advice')
    advice_error = res.get('advice_error')
    
    st.markdown("---")
    st.write("### 🥗 AI 食品營養分析結果")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【A. AI 預測營養分級】
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    color_map = {'A': '#008b45', 'B': '#85bb2f', 'C': '#fecb02', 'D': '#ee8100', 'E': '#e63e11', 'Unknown': 'gray'}
    grade_color = color_map.get(grade, 'gray')
    
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
    
    st.write(f"此食品在本系統的隨機森林分類模型中，被判定為 **{grade} 級**，屬於**{grade_desc}**的分類。")
    st.caption("ℹ️ **說明**：此分級為機器學習模型根據每 100g 營養特徵數值之預測結果，並非針對個人體質之醫療或健康評估。")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 【B. 💡 營養觀察】
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.write("#### 💡 營養觀察")
    
    if advice is not None:
        # 成功產生營養觀察：使用 markdown 渲染，保持四段式結構與排版清晰
        st.markdown(advice)
    else:
        # 營養觀察產生失敗：提供清楚友善提示與重試按鈕，不顯示未經處理的原始例外堆疊
        user_msg = advice_error.get('user_message', '營養觀察暫時無法產生，請稍後重試。') if advice_error else '營養觀察暫時無法產生，請稍後重試。'
        st.warning(f"⚠️ {user_msg}")
        
        # 允許使用者在不重新輸入數值的情況下單獨重試營養觀察
        if st.button("🔄 重新嘗試產生營養觀察", key="btn_retry_advice"):
            with st.spinner("🔄 正在重新連線產生營養觀察..."):
                nutrition_for_prompt = {
                    'energy_100g': raw_inputs['energy_kcal'],
                    'proteins_100g': raw_inputs['proteins_100g'],
                    'fat_100g': raw_inputs['fat_100g'],
                    'carbohydrates_100g': raw_inputs['carbohydrates_100g'],
                    'sugars_100g': raw_inputs['sugars_100g'],
                    'salt_100g': raw_inputs['salt_100g']
                }
                new_advice, new_error = generate_nutrition_advice(
                    nutrition_for_prompt, grade, basis=selected_basis
                )
                st.session_state.prediction_result['advice'] = new_advice
                st.session_state.prediction_result['advice_error'] = new_error
                st.rerun()
                
        if advice_error and advice_error.get('technical_details'):
            with st.expander("🛠️ 查看技術除錯細節"):
                st.caption(advice_error['technical_details'])
                
    st.markdown("---")
    st.caption("⚠️ **使用提醒**：AI 預測僅供參考，並非健康診斷；請確認營養標示與食用份量後再做判斷。")

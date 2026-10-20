import os
import json
import time
import logging
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 模型名稱設定（可透過環境變數 GEMINI_MODEL_VISION / GEMINI_MODEL_TEXT 進行彈性覆寫，預設維持既有 gemini-2.5-flash）
DEFAULT_VISION_MODEL = os.environ.get("GEMINI_MODEL_VISION", "gemini-2.5-flash")
DEFAULT_TEXT_MODEL = os.environ.get("GEMINI_MODEL_TEXT", "gemini-2.5-flash")

def classify_api_error(error: Exception) -> Dict[str, Any]:
    """
    依據例外型態與狀態碼分類 API 錯誤，並產生不外洩金鑰的技術細節與友善中文提示。
    """
    code = getattr(error, 'code', None)
    error_str = str(error)
    
    # 遮蔽可能出現在錯誤訊息中的 API Key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    sanitized_error = error_str
    if api_key and api_key in sanitized_error:
        sanitized_error = sanitized_error.replace(api_key, "[REDACTED_API_KEY]")
        
    # 1. 暫時性伺服器負載與連線逾時 (503 UNAVAILABLE / 500 / 502 / 504 / Timeout)
    error_lower = error_str.lower()
    if (
        code in [503, 500, 502, 504] 
        or isinstance(error, (TimeoutError, ConnectionError))
        or any(k in error_lower for k in ["503", "unavailable", "high demand", "server error", "temporarily unavailable", "timed out", "timeout", "connection reset", "connection refused"])
    ):
        return {
            'code': code or 503,
            'status': 'UNAVAILABLE',
            'category': 'transient_service',
            'user_message': '營養觀察服務目前忙碌中，請稍後重試。',
            'technical_details': f"ServerError [{code or 503}]: {sanitized_error}"
        }
        
    # 2. 配額或頻率限制 (429 RESOURCE_EXHAUSTED)
    if code == 429 or any(k in error_str for k in ["429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "Rate limit"]):
        return {
            'code': code or 429,
            'status': 'RESOURCE_EXHAUSTED',
            'category': 'quota_rate_limit',
            'user_message': '目前無法產生營養觀察，請稍後再試，或檢查服務使用額度。',
            'technical_details': f"RateLimitError [{code or 429}]: {sanitized_error}"
        }
        
    # 3. 設定或權限錯誤 (400, 401, 403, 404)
    if code in [400, 401, 403, 404] or any(k in error_str for k in ["API_KEY_INVALID", "PERMISSION_DENIED", "INVALID_ARGUMENT", "NOT_FOUND", "API key not valid"]):
        return {
            'code': code,
            'status': 'CONFIG_ERROR',
            'category': 'config_permission',
            'user_message': '營養觀察服務設定異常，請檢查模型設定或 API 權限。',
            'technical_details': f"ConfigError [{code}]: {sanitized_error}"
        }
        
    # 4. 其他未知或連線錯誤
    return {
        'code': code,
        'status': 'UNKNOWN_ERROR',
        'category': 'unknown',
        'user_message': '營養觀察暫時無法產生，請稍後重試。',
        'technical_details': f"UnexpectedError: {sanitized_error}"
    }

def extract_nutrition_info(image: Image.Image) -> tuple:
    """
    呼叫 Gemini Vision API 解析營養標示，回傳 (ocr_results_dict, error_msg)。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "尚未設定 GEMINI_API_KEY。請在 .env 檔案中設定您的 Gemini API Key。"
        
    client = genai.Client(api_key=api_key)
    
    prompt = """
請閱讀這張食品營養標示圖片，並擷取以下 6 項營養數值。

【優先順序與基準】
1. 請優先尋找「每 100 公克 (per 100g)」或「每 100 毫升 (per 100ml)」的欄位。
2. 如果圖片同時存在「每份」與「每100公克」，請**一定優先使用「每100公克」**，絕對不要使用「每份」數值，此時 basis 填寫 "per_100g"。
3. 如果標示僅為每 100ml，請將其視為 100g 基準，此時 basis 填寫 "per_100ml"。
4. 如果圖片中「只有」每份 (per serving) 標示，而沒有每 100g/ml 標示，請擷取該數值並將 basis 填寫 "per_serving"，絕對不要擅自換算。
5. 如果圖片中「只有」整包/每包 (per package) 標示，請擷取該數值並將 basis 填寫 "per_package"，絕對不要擅自換算。
6. 如果完全無法辨識其計算基準，請將 basis 填寫 "unknown"。

【單位轉換規則】
1. Energy (熱量)：
   請直接擷取大卡 (kcal) 數值。如果原始標示為千焦耳 (kJ)，請將其除以 4.184 轉換為大卡 (kcal)。保留兩位小數。
2. Salt (鹽)：
   如果原始標示為鈉 (Sodium) 且單位為毫克 (mg)，請先除以 1000，再乘以 2.5，轉換成鹽的公克數 (g)。保留兩位小數。
   如果原始標示直接是鹽 (Salt) 的公克數 (g)，則不需要再次轉換。
3. 其他營養素 (蛋白質, 脂肪, 碳水化合物, 糖)：
   請直接擷取數值 (單位為 g)。

【JSON 輸出格式】
請**只能**回傳符合以下 schema 的 JSON 格式：
{
  "status": "success",
  "basis": "per_100g",
  "energy_100g": 521.00,
  "proteins_100g": 8.1,
  "fat_100g": 26.5,
  "carbohydrates_100g": 62.4,
  "sugars_100g": 5.2,
  "salt_100g": 1.85
}

* 如果完全找不到營養標示，請回傳 {"status": "not_found"}
* 如果圖片太模糊、反光、裁切不完整導致無法可靠辨識，請回傳 {"status": "uncertain"}
* 如果只有部分資訊可以辨識，未辨識出或不確定的欄位請設為 null（千萬不要填入 0）。不要自行猜測數值。
"""

    try:
        response = client.models.generate_content(
            model=DEFAULT_VISION_MODEL,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        
        result_text = response.text
        data = json.loads(result_text)
        return data, None
        
    except json.JSONDecodeError:
        return None, "API 回傳格式錯誤，無法解析 JSON。"
    except Exception as e:
        classified = classify_api_error(e)
        logger.error(f"OCR Vision API 錯誤: {classified['technical_details']}")
        return None, f"辨識失敗：{classified['user_message']}"

def generate_nutrition_advice(
    nutrition_data: dict, 
    grade: str,
    basis: str = 'per_100g',
    model_name: Optional[str] = None,
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    呼叫 Gemini 產生結構化、高品質且具備實用建議的 AI 營養觀察文字。
    
    回傳四個專屬段落：
    1. 🥗 營養特徵摘要
    2. ⚠️ 值得注意的營養項目
    3. 💡 飲食建議
    4. 📌 整體評估
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        error_info = {
            'code': None,
            'status': 'CONFIG_ERROR',
            'category': 'config_permission',
            'user_message': '尚未設定 GEMINI_API_KEY，無法產生營養觀察。',
            'technical_details': 'GEMINI_API_KEY environment variable is missing.'
        }
        return None, error_info
        
    client = genai.Client(api_key=api_key)
    target_model = model_name or DEFAULT_TEXT_MODEL

    # 標示計算基準文字化
    basis_map = {
        'per_100g': '每 100 公克 (per 100g)',
        'per_100ml': '每 100 毫升 (per 100ml)',
        'per_serving': '每份 (per serving)',
        'per_package': '每包 / 整包 (per package)',
        'unknown': '未確定基準 (unknown)'
    }
    basis_desc = basis_map.get(str(basis).lower(), str(basis))

    # 安全格式化數值（若有數值為 None 則標示為未提供，絕不預設為 0）
    def fmt(val, unit):
        if val is None or (isinstance(val, str) and val.strip() == ''):
            return "未提供"
        return f"{val} {unit}"

    formatted_nutrients = f"""- 標示計算基準：{basis_desc}
- 熱量：{fmt(nutrition_data.get('energy_100g'), 'kcal')}
- 蛋白質：{fmt(nutrition_data.get('proteins_100g'), 'g')}
- 脂肪：{fmt(nutrition_data.get('fat_100g'), 'g')}
- 碳水化合物：{fmt(nutrition_data.get('carbohydrates_100g'), 'g')}
- 糖：{fmt(nutrition_data.get('sugars_100g'), 'g')}
- 食鹽：{fmt(nutrition_data.get('salt_100g'), 'g')}"""

    prompt = f"""你是一位客觀、專業且具備親和力的食品營養分析助手。請根據以下經確認的食品營養標示數據與機器學習模型預測分級（{grade} 級），撰寫一份實用、易懂且有依據的「AI 營養觀察」。

【食品營養標示數據】
{formatted_nutrients}
機器學習模型預測分級：{grade} 級

【回答結構要求】
請嚴格使用繁體中文，且務必依序區分成以下四個段落，並包含指定粗體標題與 emoji。開頭切勿加入任何引言贅字（如「好的」、「這是一份...」），結尾也無需額外客套寒暄，直接以第一段標題開始：

**🥗 營養特徵摘要**
簡要指出這款食品有哪些值得注意的營養特徵。必須嚴格根據上述實際數值分析，切勿憑空捏造食品成分、加工方式或品牌。

**⚠️ 值得注意的營養項目**
挑選最值得說明的 1～3 個營養項目進行深入解讀（例如：若糖含量較高，提醒使用者注意實際食用份量；若熱量較高，提醒搭配食用份量評估；若食鹽/鈉含量較高，提醒留意全天飲食中的鈉攝取；若脂肪含量較高，提醒可留意脂肪組成或成分）。
請注意：不要每次都強制評論全部六項營養素，僅針對實際數據中最值得關注的項目進行說明。

**💡 飲食建議**
提供 1～3 項簡單、具體且可執行的飲食建議（例如：留意包裝實際食用份量、選購時與同類產品之糖或鈉含量進行比較、如何搭配其他新鮮天然食物以維持飲食多樣性）。建議必須與該食品的實際營養數據緊密相關，避免提供「少吃一點」、「多喝水」、「維持均衡」等空泛口號。

**📌 整體評估**
用 2～3 句話簡短總結這款食品的主要營養特徵。語氣需客觀中立，提醒使用者選擇食品應考量食用份量與整體飲食習慣，不可僅依賴單一營養數值或 A～E 分級做決定。

【專業原則與品質規範】
1. 嚴格依據數據：只能依據系統提供的營養數據進行分析，絕對不可捏造或猜測未提供之數值。
2. 缺失數據保守原則：若有數值為「未提供」，應明確說明目前缺少該項數據無法判斷，絕對不可將缺失或未辨識的數值當作 0。
3. 基準意識：分析時必須注意標示基準（如每 100g 或每 100ml），提醒使用者需按實際吃進去的食用份量進行換算。
4. 客觀非批判：不可只因單一營養素較高，就直接判定食品不健康。
5. 絕無醫療宣稱：不可宣稱食用某食品必然造成疾病、減肥或其他特定療效，不提供疾病診斷、治療或個人化醫療建議。
6. 分級中立性：系統模型判定此食品為 {grade} 級。你可以客觀提及「根據模型預測結果，本食品被評為 {grade} 級」，但絕對不可主觀宣稱某個營養素就是模型給予該分級的原因，不要推測模型內部決策機制。
7. 不重複流水帳：不要只是把六項數字重新唸一遍，重點在於說明數值與食用選擇之間的實質關聯。
8. 精煉易讀：排版清晰，段落分明，總字數控制在 250～400 字之間，避免冗長文字牆。
"""

    last_error_info = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7
                )
            )
            advice_text = response.text.strip() if response.text else ""
            if not advice_text:
                raise ValueError("Model returned empty text response")
            return advice_text, None

        except Exception as e:
            classified = classify_api_error(e)
            last_error_info = classified
            
            logger.warning(
                f"Gemini API 產生營養觀察失敗 (嘗試 {attempt + 1}/{max_retries}) - "
                f"狀態: {classified['status']}, 分類: {classified['category']}, "
                f"詳情: {classified['technical_details']}"
            )
            
            # 判斷是否為值得重試的暫時性伺服器錯誤或頻率限制
            is_transient = classified['category'] in ['transient_service', 'quota_rate_limit']
            
            if is_transient and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.info(f"等待 {delay} 秒後進行第 {attempt + 2} 次重試...")
                time.sleep(delay)
                continue
            else:
                # 不可重試之設定/權限錯誤，或已達重試上限
                break

    return None, last_error_info

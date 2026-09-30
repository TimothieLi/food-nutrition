import os
import json
from PIL import Image
from google import genai
from google.genai import types

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
請優先尋找「每 100 公克 (per 100g)」或「每 100 毫升 (per 100ml)」的欄位。
如果圖片同時存在「每份」與「每100公克」，請**一定優先使用「每100公克」**，絕對不要使用「每份」數值。
如果標示為每 100ml，請將其視為 100g 基準。

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
* 如果只有部分資訊可以辨識，未辨識出的欄位請設為 null。不要自行猜測數值。
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
        return None, f"API 呼叫失敗或網路錯誤：{str(e)}"

def generate_nutrition_advice(nutrition_data: dict, grade: str) -> str:
    """
    呼叫 Gemini 產生一段簡單、易懂的營養觀察文字。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "尚未設定 GEMINI_API_KEY，無法產生營養觀察。"
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
請根據以下食品的每 100g 營養資訊與 AI 預測分級（{grade} 級），撰寫一段「人性化、像營養專家在向一般民眾解釋」的營養觀察。

營養資訊：
- 熱量：{nutrition_data['energy_100g']} kcal/100g
- 蛋白質：{nutrition_data['proteins_100g']} g/100g
- 脂肪：{nutrition_data['fat_100g']} g/100g
- 碳水化合物：{nutrition_data['carbohydrates_100g']} g/100g
- 糖：{nutrition_data['sugars_100g']} g/100g
- 食鹽：{nutrition_data['salt_100g']} g/100g

寫作原則：
1. 語氣需專業、客觀且精煉，直接切入重點。絕對不要使用「哈囉」、「各位朋友」等口語招呼語，也不要有過度情緒化的字眼。
2. 重點放在「客觀解讀這份數據，並提醒一般大眾在飲食上應該注意什麼」。
3. 如果各項營養素皆為 0（例如零卡飲料或水），請直接指出這是幾乎不含熱量及必需營養素的食品，無需過度誇飾其「零負擔」。
4. 不要只是把數字重新列出來。
5. 不要自行創造不存在的營養數據。
6. 不要將所有食品都直接稱為「高糖」、「高脂肪」，必須要有合理的數值依據。
7. 不要做疾病、醫療或治療建議。
8. 不要聲稱自己是營養師或醫療專業人員。
9. 字數控制在 80~120 字以內，精確扼要，分段清楚。
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
        return response.text
    except Exception as e:
        return f"無法產生營養觀察：{str(e)}"

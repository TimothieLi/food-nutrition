import re
import pytesseract
from PIL import Image

def extract_nutrition_info(image: Image.Image) -> dict:
    """
    使用 PyTesseract 進行 OCR，並使用 Regex 擷取 6 項營養數值。
    回傳字典格式，找不到數值則對應 key 為 None。
    """
    results = {
        'energy_100g': None,
        'proteins_100g': None,
        'fat_100g': None,
        'carbohydrates_100g': None,
        'sugars_100g': None,
        'salt_100g': None
    }
    
    try:
        # 將影像轉為灰階以增強 OCR 效果
        gray_image = image.convert('L')
        text = pytesseract.image_to_string(gray_image, lang='eng')
        text = text.lower()
    except Exception as e:
        # OCR 引擎發生錯誤 (如未安裝 Tesseract)
        print(f"OCR Error: {e}")
        return results

    if not text.strip():
        return results

    # 定義各特徵的常見關鍵字 regex (允許少許拼寫錯誤或標點)
    patterns = {
        'energy_100g': r'(?:energy|calories|kcal)\s*[:\-]?\s*(\d+(?:\.\d+)?)',
        'proteins_100g': r'protein[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)',
        'fat_100g': r'(?:fat|lipid)[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)',
        'carbohydrates_100g': r'carbohydrate[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)',
        'sugars_100g': r'sugar[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)',
        'salt_100g': r'(?:salt|sodium)\s*[:\-]?\s*(\d+(?:\.\d+)?)'
    }

    # 嘗試從 OCR 文本中找出符合的數值
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            try:
                results[key] = float(match.group(1))
            except ValueError:
                pass

    return results

def is_nutrition_info_found(info: dict) -> bool:
    """
    檢查是否完全找不到任何營養標示。
    如果全都是 None，回傳 False。
    """
    return any(value is not None for value in info.values())

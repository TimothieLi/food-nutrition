import math
from typing import Dict, Any, List, Optional

FIELD_NAMES = {
    'energy_kcal': '熱量 (kcal)',
    'proteins_100g': '蛋白質 (g)',
    'fat_100g': '脂肪 (g)',
    'carbohydrates_100g': '碳水化合物 (g)',
    'sugars_100g': '糖 (g)',
    'salt_100g': '食鹽 (g)'
}

NUTRITION_KEYS = [
    'energy_kcal',
    'proteins_100g',
    'fat_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'salt_100g'
]

BASIS_DISPLAY_NAMES = {
    'per_100g': '每 100 公克 (per 100g)',
    'per_100ml': '每 100 毫升 (per 100ml)',
    'per_serving': '每份 (per serving)',
    'per_package': '每包 / 整包 (per package)',
    'unknown': '無法確認基準 (unknown)'
}

def validate_nutrition_data(
    data: Dict[str, Any],
    basis: str = 'per_100g',
    confirmed_all_zero: bool = False
) -> Dict[str, Any]:
    """
    驗證輸入的食品營養成分資料。
    
    回傳字典結構：
    {
        'status': 'valid' | 'warning' | 'invalid',
        'is_valid': bool,           # 唯有 True 時才允許進入模型預測
        'errors': List[str],        # 致命錯誤（阻擋預測）
        'warnings': List[str],      # 警示訊息（需使用者確認）
        'cleaned_data': dict|None,  # 轉換為模型所需的 kJ 與克數格式
        'info': dict                # 附帶狀態資訊
    }
    """
    errors: List[str] = []
    warnings: List[str] = []
    cleaned_values: Dict[str, float] = {}
    
    # ----------------------------------------------------
    # 1. 數值存在性與格式轉換檢查 (缺失值、空字串、NaN、Inf、格式錯誤)
    # ----------------------------------------------------
    missing_fields: List[str] = []
    
    for key in NUTRITION_KEYS:
        field_label = FIELD_NAMES[key]
        raw_val = data.get(key)
        
        # 缺失值檢查 (None 或空字串)
        if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ''):
            missing_fields.append(field_label)
            continue
            
        # 嘗試轉換成 float
        try:
            val_float = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"【{field_label}】內容「{raw_val}」無法轉換為有效數字，請重新輸入。")
            continue
            
        # 檢查 NaN 與 無限大
        if math.isnan(val_float):
            errors.append(f"【{field_label}】數值無效 (NaN)，請填寫有效數值。")
            continue
        if math.isinf(val_float):
            errors.append(f"【{field_label}】數值為正負無限大 (Inf)，請填寫有效數值。")
            continue
            
        # 負數檢查
        if val_float < 0:
            errors.append(f"【{field_label}】數值不可為負數（目前輸入：{val_float}）。")
            continue
            
        cleaned_values[key] = val_float

    if missing_fields:
        errors.append(f"缺少必要欄位數值：{ '、'.join(missing_fields) }。請補齊所有營養資訊後再行分析。")

    # 若基礎格式已有錯誤或缺漏，直接判定為 invalid，阻止後續進入模型
    if errors:
        return {
            'status': 'invalid',
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_data': None,
            'info': {'missing_fields': missing_fields}
        }

    # ----------------------------------------------------
    # 2. 單位與數值基準檢查
    # ----------------------------------------------------
    basis_str = str(basis).strip().lower()
    if basis_str not in ['per_100g', 'per_100ml']:
        display_basis = BASIS_DISPLAY_NAMES.get(basis_str, basis_str)
        errors.append(
            f"【標示基準不符】目前標示基準為「{display_basis}」。"
            f"本預測模型是以「每 100 公克 (per 100g)」為標準特徵，請手動換算為每 100g 數值並選擇每 100 公克基準。"
        )
        return {
            'status': 'invalid',
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_data': None,
            'info': {'basis': basis_str}
        }

    # ----------------------------------------------------
    # 3. 六項數值全部為 0 的檢查
    # ----------------------------------------------------
    all_zero = all(abs(cleaned_values[k]) < 1e-9 for k in NUTRITION_KEYS)
    if all_zero:
        if not confirmed_all_zero:
            # 尚未經過人工特別確認
            warnings.append("目前六項營養數值皆為 0，請確認 OCR 是否正確辨識營養標示，或手動檢查輸入數值。")
            return {
                'status': 'warning',
                'is_valid': False,
                'errors': [],
                'warnings': warnings,
                'cleaned_data': None,
                'info': {'all_zero': True, 'requires_confirmation': True}
            }
        else:
            # 使用者已確認為全零食品（如包裝水、無糖無熱量飲品）
            model_input = {
                'energy_100g': 0.0,
                'proteins_100g': 0.0,
                'fat_100g': 0.0,
                'carbohydrates_100g': 0.0,
                'sugars_100g': 0.0,
                'salt_100g': 0.0
            }
            return {
                'status': 'valid',
                'is_valid': True,
                'errors': [],
                'warnings': warnings,
                'cleaned_data': model_input,
                'info': {'all_zero': True, 'confirmed_all_zero': True}
            }

    # ----------------------------------------------------
    # 4. 營養數值合理性檢查 (物理與生理學限制)
    # ----------------------------------------------------
    energy_kcal = cleaned_values['energy_kcal']
    protein = cleaned_values['proteins_100g']
    fat = cleaned_values['fat_100g']
    carbs = cleaned_values['carbohydrates_100g']
    sugars = cleaned_values['sugars_100g']
    salt = cleaned_values['salt_100g']

    # (1) 糖含量 vs. 碳水化合物含量 (糖為碳水化合物之一部分)
    if sugars > carbs:
        errors.append(
            f"【糖與碳水化合物數值不合理】糖含量（{sugars} g）高於碳水化合物含量（{carbs} g）。"
            f"一般情況下糖為碳水化合物的一部分，請確認原始營養標示之數值與單位。"
        )

    # (2) 極端數值檢查：單項營養素不可超過每 100g 食品重量 (100g)
    single_nutrients = [
        ('proteins_100g', protein),
        ('fat_100g', fat),
        ('carbohydrates_100g', carbs),
        ('sugars_100g', sugars),
        ('salt_100g', salt)
    ]
    for key, val in single_nutrients:
        if val > 100.0:
            errors.append(
                f"【{FIELD_NAMES[key]} 數值異常】{FIELD_NAMES[key]}（{val} g/100g）超過每 100g 食品物理上限（100g）。"
                f"請確認是否誤填整包數值或單位錯誤。"
            )

    # (3) 極端熱量檢查：100g 純脂肪熱量約為 900 kcal，超過 1000 kcal 屬於物理不可能
    if energy_kcal > 1000.0:
        errors.append(
            f"【熱量極端異常】熱量（{energy_kcal} kcal/100g）超過每 100g 食品物理極限（純油脂約 900 kcal）。"
            f"請確認是否誤填千焦耳 (kJ) 或整包總熱量。"
        )

    # (4) 巨量營養素與食鹽重量總和檢查 (容許 105g 包含分析誤差與小數進位)
    macro_sum = protein + fat + carbs + salt
    if macro_sum > 105.0:
        errors.append(
            f"【巨量營養素總和過高】蛋白質 + 脂肪 + 碳水化合物 + 食鹽 之總和為 {round(macro_sum, 2)} g，"
            f"超過每 100g 食品重量總額（容許合理誤差至 105g）。請檢查各項營養素數值。"
        )

    if errors:
        return {
            'status': 'invalid',
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_data': None,
            'info': {'all_zero': False}
        }

    # ----------------------------------------------------
    # 5. 驗證通過：整理模型特徵資料 (將 kcal 轉為 kJ)
    # ----------------------------------------------------
    model_input = {
        'energy_100g': energy_kcal * 4.184,  # 模型訓練特徵為 kJ
        'proteins_100g': protein,
        'fat_100g': fat,
        'carbohydrates_100g': carbs,
        'sugars_100g': sugars,
        'salt_100g': salt
    }

    return {
        'status': 'valid',
        'is_valid': True,
        'errors': [],
        'warnings': warnings,
        'cleaned_data': model_input,
        'info': {'all_zero': False}
    }

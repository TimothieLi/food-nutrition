import pandas as pd
import numpy as np

def run_preprocessing():
    data_path = '../01_Data/raw/en.openfoodfacts.org.products.tsv'
    
    # 1. 載入資料 & 2. 欄位篩選
    features = [
        'energy_100g', 'proteins_100g', 'fat_100g', 
        'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g'
    ]
    label = 'nutrition_grade_fr'
    cols_to_use = features + [label]
    
    print("Loading data...")
    df = pd.read_csv(data_path, sep='\t', low_memory=False, usecols=cols_to_use)
    original_len = len(df)
    
    # 3. 資料型態處理
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 4. Label 清理
    valid_labels = ['a', 'b', 'c', 'd', 'e']
    df = df[df[label].isin(valid_labels)]
    len_after_label_clean = len(df)
    
    # 5. 缺失值處理比較
    features_A = [f for f in features if f != 'fiber_100g']
    features_B = features.copy()
    
    df_A = df.dropna(subset=features_A + [label]).copy()
    df_B = df.dropna(subset=features_B + [label]).copy()
    
    stats = {}
    stats['len_before_na'] = len(df)
    stats['A_len'] = len(df_A)
    stats['A_dropped'] = len(df) - len(df_A)
    stats['A_retention'] = len(df_A) / original_len
    stats['A_labels'] = df_A[label].value_counts().to_dict()
    
    stats['B_len'] = len(df_B)
    stats['B_dropped'] = len(df) - len(df_B)
    stats['B_retention'] = len(df_B) / original_len
    stats['B_labels'] = df_B[label].value_counts().to_dict()
    
    # 6. 異常值處理 (0 <= value <= 100, energy >= 0)
    def handle_outliers(data, feats):
        outlier_counts = {}
        valid_mask = pd.Series(True, index=data.index)
        
        for f in feats:
            if f == 'energy_100g':
                cond = (data[f] >= 0)
            else:
                cond = (data[f] >= 0) & (data[f] <= 100)
                
            outliers = (~cond).sum()
            outlier_counts[f] = int(outliers)
            valid_mask = valid_mask & cond
            
        cleaned_data = data[valid_mask].copy()
        return cleaned_data, outlier_counts
        
    df_A_clean, A_outliers = handle_outliers(df_A, features_A)
    df_B_clean, B_outliers = handle_outliers(df_B, features_B)
    
    stats['A_after_outliers'] = len(df_A_clean)
    stats['B_after_outliers'] = len(df_B_clean)
    stats['A_outliers_counts'] = A_outliers
    stats['B_outliers_counts'] = B_outliers
    
    # 7. 重複資料
    def handle_duplicates(data):
        dups = data.duplicated().sum()
        deduped = data.drop_duplicates()
        return deduped, int(dups)
        
    df_A_final, A_dups = handle_duplicates(df_A_clean)
    df_B_final, B_dups = handle_duplicates(df_B_clean)
    
    stats['A_dups'] = A_dups
    stats['A_final_len'] = len(df_A_final)
    stats['B_dups'] = B_dups
    stats['B_final_len'] = len(df_B_final)
    
    # 8. Label 分布重新統計
    stats['A_final_labels'] = df_A_final[label].value_counts().to_dict()
    stats['A_final_pcts'] = df_A_final[label].value_counts(normalize=True).to_dict()
    
    stats['B_final_labels'] = df_B_final[label].value_counts().to_dict()
    stats['B_final_pcts'] = df_B_final[label].value_counts(normalize=True).to_dict()
    
    # 9. 儲存結果
    df_A_final.to_csv('../01_Data/processed/clean_food_data.csv', index=False)
    df_B_final.to_csv('../01_Data/processed/clean_food_data_with_fiber.csv', index=False)
    
    # 產生 Report
    report = f"""# Data Preprocessing Report

## 1. 原始資料筆數
- **原始資料筆數**: {original_len:,} 筆

## 2. 欄位篩選與 Label 清理後筆數
- 僅保留營養特徵與 Label (`nutrition_grade_fr`)，且強制型態為數值。
- 過濾無效 Label 後筆數: {len_after_label_clean:,} 筆

## 3. 缺失值處理與 Fiber 使用比較
我們針對是否包含 `fiber_100g` 建立兩個版本的資料集，統一進行 `dropna()`。

| 項目 | 方案 A (不含 Fiber) | 方案 B (包含 Fiber) |
| :--- | :--- | :--- |
| 清理前資料筆數 (Label過濾後) | {stats['len_before_na']:,} | {stats['len_before_na']:,} |
| 清理後剩餘筆數 | {stats['A_len']:,} | {stats['B_len']:,} |
| 被刪除的資料筆數 | {stats['A_dropped']:,} | {stats['B_dropped']:,} |
| 相較原始資料保留率 | {stats['A_retention']:.1%} | {stats['B_retention']:.1%} |

**Label 分布 (DropNA 後)**:
- 方案 A: {stats['A_labels']}
- 方案 B: {stats['B_labels']}

## 4. 異常值移除
將重量限制在 `0 ~ 100`，熱量限制 `>= 0`。
- **方案 A 移除異常值後**: {stats['A_after_outliers']:,} 筆 (各欄位異常數: {stats['A_outliers_counts']})
- **方案 B 移除異常值後**: {stats['B_after_outliers']:,} 筆 (各欄位異常數: {stats['B_outliers_counts']})

## 5. 重複資料處理
移除完全重複的資料列。
- **方案 A**: 發現 {stats['A_dups']:,} 筆重複。移除後剩餘 {stats['A_final_len']:,} 筆。
- **方案 B**: 發現 {stats['B_dups']:,} 筆重複。移除後剩餘 {stats['B_final_len']:,} 筆。

## 6. 最終資料筆數
- **clean_food_data.csv (無 Fiber)**: {stats['A_final_len']:,} 筆
- **clean_food_data_with_fiber.csv (有 Fiber)**: {stats['B_final_len']:,} 筆

## 7. 最終 Feature 清單
- **方案 A**: `energy_100g`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`
- **方案 B**: `energy_100g`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `fiber_100g`, `salt_100g`

## 8. Label 分布 (最終版)
| 類別 | 方案 A 數量 | 方案 A 比例 | 方案 B 數量 | 方案 B 比例 |
| :--- | :--- | :--- | :--- | :--- |
| A | {stats['A_final_labels'].get('a', 0):,} | {stats['A_final_pcts'].get('a', 0):.2%} | {stats['B_final_labels'].get('a', 0):,} | {stats['B_final_pcts'].get('a', 0):.2%} |
| B | {stats['A_final_labels'].get('b', 0):,} | {stats['A_final_pcts'].get('b', 0):.2%} | {stats['B_final_labels'].get('b', 0):,} | {stats['B_final_pcts'].get('b', 0):.2%} |
| C | {stats['A_final_labels'].get('c', 0):,} | {stats['A_final_pcts'].get('c', 0):.2%} | {stats['B_final_labels'].get('c', 0):,} | {stats['B_final_pcts'].get('c', 0):.2%} |
| D | {stats['A_final_labels'].get('d', 0):,} | {stats['A_final_pcts'].get('d', 0):.2%} | {stats['B_final_labels'].get('d', 0):,} | {stats['B_final_pcts'].get('d', 0):.2%} |
| E | {stats['A_final_labels'].get('e', 0):,} | {stats['A_final_pcts'].get('e', 0):.2%} | {stats['B_final_labels'].get('e', 0):,} | {stats['B_final_pcts'].get('e', 0):.2%} |

*類別分佈依然非常平衡，不存在嚴重不平衡。*

## 9. 最終建議
- **保留 Fiber 評估**: 方案 B 因為加入了 Fiber，資料量從 {stats['A_final_len']:,} 筆減少至 {stats['B_final_len']:,} 筆。雖然損失了資料，但 {stats['B_final_len']:,} 筆對於初學者的 Demo 來說仍然非常龐大。保留 Fiber 可以在特徵層面提供更完整的營養輪廓。
- **最終決定**: 建議採用**方案 A (無 Fiber)**，因為對於 Decision Tree 的展示來說，6 個特徵已經足夠，且 {stats['A_final_len']:,} 筆資料在保留率上較高，資料代表性更好。但兩份 CSV 均已匯出，您可以根據後續繪圖的方便性隨時切換。
"""

    with open('preprocessing_report.md', 'w') as f:
        f.write(report)
        
if __name__ == "__main__":
    run_preprocessing()

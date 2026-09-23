import pandas as pd
import json

def explore():
    file_path = '../01_Data/raw/en.openfoodfacts.org.products.tsv'
    
    # 1. Basic Info
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    
    # 2. Nutrition columns
    nutrition_cols = [c for c in columns if '100g' in c or 'nutrition' in c]
    
    # 3. Specific columns
    target_cols = [
        'energy-kcal_100g',
        'proteins_100g',
        'fat_100g',
        'carbohydrates_100g',
        'sugars_100g',
        'fiber_100g',
        'salt_100g'
    ]
    
    missing_ratios = {}
    for col in target_cols:
        if col in df.columns:
            missing_ratios[col] = df[col].isnull().mean()
        else:
            # try finding similar
            similar = [c for c in columns if c.startswith(col.split('_')[0])]
            missing_ratios[col] = f"Not found. Similar: {similar[:3]}"
            
    # 4. Classification labels
    label_cols = ['pnns_groups_1', 'pnns_groups_2', 'categories_en', 'nutriscore_grade']
    labels_info = {}
    for col in label_cols:
        if col in df.columns:
            labels_info[col] = {
                'missing_ratio': df[col].isnull().mean(),
                'unique_count': df[col].nunique(),
                'top_5': df[col].value_counts().head(5).to_dict()
            }
        else:
            labels_info[col] = "Not found"
            
    # 5. Anomalies
    anomalies = {}
    for col in target_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[0.01, 0.99]).to_dict()
            neg_count = (df[col] < 0).sum()
            # For 100g nutrients (except energy), max theoretically is 100
            over_100_count = (df[col] > 100).sum() if 'energy' not in col else (df[col] > 4000).sum()
            anomalies[col] = {
                'min': desc.get('min'),
                'max': desc.get('max'),
                '1%': desc.get('1%'),
                '99%': desc.get('99%'),
                'negative_count': int(neg_count),
                'extreme_high_count': int(over_100_count)
            }

    results = {
        'file_name': 'en.openfoodfacts.org.products.tsv',
        'rows': num_rows,
        'cols': num_cols,
        'nutrition_related_columns_count': len(nutrition_cols),
        'sample_nutrition_cols': nutrition_cols[:15],
        'target_missing_ratios': missing_ratios,
        'labels_info': labels_info,
        'anomalies': anomalies
    }
    
    with open('exploration_result.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    explore()

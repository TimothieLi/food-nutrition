import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def run_eda():
    data_path = '../01_Data/raw/en.openfoodfacts.org.products.tsv'
    output_dir = '.'
    fig_dir = './figures'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    features = [
        'energy_100g', 'proteins_100g', 'fat_100g',
        'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g'
    ]
    labels = ['nutrition_grade_fr', 'pnns_groups_1']
    
    print("Loading data...")
    df = pd.read_csv(data_path, sep='\t', low_memory=False, usecols=features + labels)
    
    stats = {}
    
    # 1 & 2: Feature Statistics & Distributions
    print("Analyzing features...")
    feature_stats = {}
    for col in features:
        # Calculate stats on raw data
        col_data = df[col]
        missing_ratio = col_data.isnull().mean()
        
        # Outliers threshold for plotting
        if 'energy' in col:
            valid_max = 4000
        else:
            valid_max = 100
            
        extreme_outliers = (col_data < 0).sum() + (col_data > valid_max).sum()
        
        feature_stats[col] = {
            'mean': float(col_data.mean()) if pd.notnull(col_data.mean()) else None,
            'median': float(col_data.median()) if pd.notnull(col_data.median()) else None,
            'min': float(col_data.min()) if pd.notnull(col_data.min()) else None,
            'max': float(col_data.max()) if pd.notnull(col_data.max()) else None,
            'missing_ratio': float(missing_ratio),
            'extreme_outliers_count': int(extreme_outliers)
        }
        
        # Plot distribution (filtered for readable plots)
        plt.figure(figsize=(8, 5))
        valid_data = col_data[(col_data >= 0) & (col_data <= valid_max)].dropna()
        sns.histplot(valid_data, bins=50, kde=True)
        plt.title(f'Distribution of {col} (0-{valid_max} range)')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'dist_{col}.png'))
        plt.close()
        
    stats['feature_stats'] = feature_stats
    
    # 3: nutrition_grade_fr Analysis
    print("Analyzing nutrition_grade_fr...")
    grade_counts = df['nutrition_grade_fr'].value_counts(dropna=False)
    grade_percentages = df['nutrition_grade_fr'].value_counts(normalize=True, dropna=False)
    
    stats['nutrition_grade_fr'] = {
        'counts': grade_counts.to_dict(),
        'percentages': grade_percentages.to_dict(),
        'missing_ratio': float(df['nutrition_grade_fr'].isnull().mean())
    }
    
    plt.figure(figsize=(8, 5))
    grade_counts.drop(np.nan, errors='ignore').sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution of nutrition_grade_fr')
    plt.xlabel('Nutri-Score Grade')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dist_nutrition_grade_fr.png'))
    plt.close()
    
    # 4: pnns_groups_1 Analysis
    print("Analyzing pnns_groups_1...")
    pnns_counts = df['pnns_groups_1'].value_counts(dropna=False)
    pnns_percentages = df['pnns_groups_1'].value_counts(normalize=True, dropna=False)
    
    stats['pnns_groups_1'] = {
        'counts': pnns_counts.to_dict(),
        'percentages': pnns_percentages.to_dict(),
        'missing_ratio': float(df['pnns_groups_1'].isnull().mean()),
        'unknown_ratio': float((df['pnns_groups_1'] == 'unknown').mean())
    }
    
    plt.figure(figsize=(10, 8))
    pnns_counts.head(15).plot(kind='barh', color='lightgreen')
    plt.title('Top 15 Categories in pnns_groups_1')
    plt.xlabel('Count')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'dist_pnns_groups_1.png'))
    plt.close()
    
    # 5: Correlation Matrix
    print("Analyzing correlation...")
    # Filter valid rows for correlation to avoid extreme outliers skewing the metric
    corr_df = df[features].copy()
    for col in features:
        if 'energy' in col:
            corr_df.loc[(corr_df[col] < 0) | (corr_df[col] > 4000), col] = np.nan
        else:
            corr_df.loc[(corr_df[col] < 0) | (corr_df[col] > 100), col] = np.nan
            
    corr_matrix = corr_df.corr()
    stats['correlation'] = corr_matrix.to_dict()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Nutrition Features')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Save stats
    with open(os.path.join(output_dir, 'eda_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("EDA completed successfully.")

if __name__ == '__main__':
    run_eda()

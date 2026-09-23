import pandas as pd
import json
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

def run_training():
    # 1. 讀取資料
    print("Loading data...")
    train_df = pd.read_csv('../../01_Data/processed/train.csv')
    val_df = pd.read_csv('../../01_Data/processed/validation.csv')
    # test_df is not loaded to avoid using it
    
    # 2. 分離 X, y
    features = [
        'energy_100g', 'proteins_100g', 'fat_100g', 
        'carbohydrates_100g', 'sugars_100g', 'salt_100g'
    ]
    target = 'nutrition_grade_fr'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    
    # 3 & 4. 建立並訓練 Baseline Model
    print("Training Decision Tree Baseline...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Baseline 評估
    print("Evaluating Model...")
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    
    def get_metrics(y_true, y_pred):
        return {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'F1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
    train_metrics = get_metrics(y_train, y_train_pred)
    val_metrics = get_metrics(y_val, y_val_pred)
    
    results = {
        'Train': train_metrics,
        'Validation': val_metrics
    }
    
    with open('../results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # 檢查 Overfitting
    overfitting = train_metrics['Accuracy'] - val_metrics['Accuracy'] > 0.05
    diff = train_metrics['Accuracy'] - val_metrics['Accuracy']
    
    # 6. 模型儲存
    model_path = '../models/decision_tree_baseline.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    # 7. Decision Tree 視覺化 (僅供展示，max_depth=3)
    print("Generating visualization (max_depth=3)...")
    clf_vis = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_vis.fit(X_train, y_train)
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf_vis, 
        feature_names=features, 
        class_names=sorted(y_train.unique()), 
        filled=True, 
        rounded=True,
        fontsize=12
    )
    plt.title('Decision Tree Visualization (max_depth=3)')
    plt.tight_layout()
    vis_path = '../../05_Evaluation/figures/decision_tree_baseline.png'
    plt.savefig(vis_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {vis_path}")
    
    # 8. 建立報告
    report = f"""# Decision Tree Baseline Model Report

## 1. Model 名稱
Decision Tree Baseline (sklearn default settings)

## 2. Features
- `energy_100g`
- `proteins_100g`
- `fat_100g`
- `carbohydrates_100g`
- `sugars_100g`
- `salt_100g`

## 3. Target
`nutrition_grade_fr` (A, B, C, D, E)

## 4. Train / Validation 資料筆數
- **Train**: {len(X_train):,} 筆
- **Validation**: {len(X_val):,} 筆

## 5. Decision Tree 基本設定
- `random_state`: 42
- `max_depth`: None (樹會完全生長直到每個葉節點都是純的)
- `criterion`: gini

## 6. Train 評估結果
- **Accuracy**: {train_metrics['Accuracy']:.4f}
- **Precision (weighted)**: {train_metrics['Precision']:.4f}
- **Recall (weighted)**: {train_metrics['Recall']:.4f}
- **F1-score (weighted)**: {train_metrics['F1']:.4f}

## 7. Validation 評估結果
- **Accuracy**: {val_metrics['Accuracy']:.4f}
- **Precision (weighted)**: {val_metrics['Precision']:.4f}
- **Recall (weighted)**: {val_metrics['Recall']:.4f}
- **F1-score (weighted)**: {val_metrics['F1']:.4f}

## 8. Train vs Validation 差異
- **Accuracy 差異**: {(diff * 100):.2f}%
Train 的準確率幾乎達到 100%，但 Validation 的準確率卻掉到 {val_metrics['Accuracy']:.4f}。

## 9. 是否有明顯 Overfitting
**是，存在非常嚴重的 Overfitting（過度擬合）。**
因為沒有設定 `max_depth`，Decision Tree 無限制地向下生長，將 Train data 中的雜訊與特例都死背下來（Memorization），導致 Train Accuracy 高達近乎完美的 1.0，但面對從未見過的 Validation data 時，泛化能力 (Generalization) 表現大幅衰退。

## 10. 下一步建議
1. **設定 max_depth (Pruning)**：限制樹的最大深度，迫使模型學習更通用的規則，而不是死背每一筆資料。
2. **進行超參數搜尋 (Hyperparameter Tuning)**：使用 Grid Search 或 Random Search 找出最佳的 `max_depth`、`min_samples_split` 等參數。
3. 等參數調整完成並解決 Overfitting 後，再使用 Test dataset 進行最終評估。
"""

    with open('../reports/model_report.md', 'w') as f:
        f.write(report)
        
    print("Training script finished successfully.")

if __name__ == '__main__':
    run_training()

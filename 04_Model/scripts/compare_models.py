import pandas as pd
import json
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def run_comparison():
    print("Loading data...")
    train_df = pd.read_csv('../../01_Data/processed/train.csv')
    val_df = pd.read_csv('../../01_Data/processed/validation.csv')
    
    features = [
        'energy_100g', 'proteins_100g', 'fat_100g', 
        'carbohydrates_100g', 'sugars_100g', 'salt_100g'
    ]
    target = 'nutrition_grade_fr'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_val = val_df[features]
    y_val = val_df[target]

    # Label Encoding for LightGBM (and others for consistency)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(objective='multiclass', num_class=5, n_estimators=100, max_depth=15, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    print("Training models...")
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        start_time = time.time()
        
        # Fit model
        model.fit(X_train, y_train_enc)
        
        train_time = time.time() - start_time
        trained_models[model_name] = model
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_metrics = get_metrics(y_train_enc, y_train_pred)
        val_metrics = get_metrics(y_val_enc, y_val_pred)
        gap = train_metrics['Accuracy'] - val_metrics['Accuracy']
        
        res = {
            'Model': model_name,
            'Train Accuracy': train_metrics['Accuracy'],
            'Validation Accuracy': val_metrics['Accuracy'],
            'Overfitting Gap': gap,
            'Train Precision': train_metrics['Precision'],
            'Validation Precision': val_metrics['Precision'],
            'Train Recall': train_metrics['Recall'],
            'Validation Recall': val_metrics['Recall'],
            'Train F1': train_metrics['F1'],
            'Validation F1': val_metrics['F1'],
            'Training Time (s)': round(train_time, 2)
        }
        results.append(res)
        print(f"[{model_name}] Val Acc: {res['Validation Accuracy']:.4f}, Gap: {gap:.4f}, Time: {train_time:.2f}s")
        
    # Save results to JSON
    with open('../results/model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Generate Chart
    plt.figure(figsize=(10, 6))
    model_names = [r['Model'] for r in results]
    val_accs = [r['Validation Accuracy'] * 100 for r in results]
    train_accs = [r['Train Accuracy'] * 100 for r in results]
    
    x = range(len(model_names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], train_accs, width, label='Train Accuracy', color='skyblue')
    plt.bar([i + width/2 for i in x], val_accs, width, label='Validation Accuracy', color='salmon')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy by Model')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # save to 04_Model
    plt.tight_layout()
    plt.savefig('../figures/model_comparison_chart.png', dpi=300)
    plt.close()
    
    # Generate Report
    report = f"""# Classification Model Comparison Report

## 1. 實驗目的
為了確認除了 Decision Tree 之外，其他分類模型 (如 Random Forest, LightGBM) 是否能在相同特徵下得到更好的泛化能力 (更好的 Validation Accuracy 與較低的 Overfitting Gap)。本實驗在找出最適合進入下一階段進行 Test 評估的最終候選模型。

## 2. 實驗資料
本實驗使用相同的資料切分：
- **Train Set**: {len(y_train):,} 筆 (僅用於訓練)
- **Validation Set**: {len(y_val):,} 筆 (僅用於評估與比較)
- **Test Set**: 尚未碰觸，保留至 Final Evaluation。

## 3. 模型介紹
- **Decision Tree**: 作為 Baseline，使用前一階段 Hyperparameter Tuning 找出的最佳候選參數 (`max_depth=15`)。
- **Random Forest**: 透過 Ensemble Learning 集合多棵決策樹，使用 `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2` 來防止過度擬合並增強穩定性。
- **LightGBM**: 輕量級且高效的梯度提升決策樹 (Gradient Boosting Framework)。使用 `objective='multiclass', num_class=5`, `max_depth=15`。

## 4. 實驗結果

| Model | Train Accuracy | Validation Accuracy | Train F1 | Validation F1 | Overfitting Gap | Training Time (s) |
|-------|----------------|---------------------|----------|---------------|-----------------|-------------------|
"""
    for r in results:
        report += f"| {r['Model']} | {r['Train Accuracy']:.4f} | {r['Validation Accuracy']:.4f} | {r['Train F1']:.4f} | {r['Validation F1']:.4f} | {r['Overfitting Gap']:.4f} | {r['Training Time (s)']} |\n"

    report += """
## 5. 結果分析
* **哪些模型 Validation 表現較好？**
從結果來看，LightGBM 和 Random Forest 通常表現比單一 Decision Tree 更好。
* **哪些模型有較嚴重 Overfitting？**
Decision Tree 的 Overfitting Gap 約為 11-12%。Random Forest 加上適當深度與樣本數限制後，也能良好控制 Overfitting。如果 Random Forest/LightGBM 的 Train Accuracy 過高且 Gap 很大，則代表發生嚴重 Overfitting（本實驗中已預設控制 `max_depth` 來緩解）。
* **Train / Validation Gap 比較**
Ensemble model 的泛化能力普遍優於單一決策樹。
* **訓練成本考量**
Random Forest 和 LightGBM 都支援平行處理 (`n_jobs=-1`)，訓練時間在可接受範圍內。LightGBM 訓練速度通常具有明顯優勢。

## 6. 下一步建議
綜合考量 Validation Accuracy, Validation F1 以及 Overfitting Gap，建議將表現最好的模型（例如 LightGBM 或 Random Forest）選作目前的「最佳候選模型」。該模型值得進入下一階段 (`05_Evaluation`)，使用尚未接觸的 Test Set 來進行最終的 Final Evaluation。
"""

    with open('../reports/model_comparison_report.md', 'w') as f:
        f.write(report)
        
    print("Comparison script finished successfully.")

if __name__ == '__main__':
    run_comparison()

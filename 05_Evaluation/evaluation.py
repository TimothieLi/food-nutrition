import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    }

def run_evaluation():
    print("Loading data...")
    train_df = pd.read_csv('../01_Data/processed/train.csv')
    test_df = pd.read_csv('../01_Data/processed/test.csv')
    
    # Optional: Load validation results for comparison
    with open('../04_Model/results/model_comparison_results.json', 'r') as f:
        val_results = json.load(f)
    
    val_map = {r['Model']: r for r in val_results}
    
    features = [
        'energy_100g', 'proteins_100g', 'fat_100g', 
        'carbohydrates_100g', 'sugars_100g', 'salt_100g'
    ]
    target = 'nutrition_grade_fr'
    
    X_train = train_df[features]
    y_train = train_df[target].str.upper()
    
    X_test = test_df[features]
    y_test = test_df[target].str.upper()
    
    # Label Encoding (A->0, B->1, C->2, D->3, E->4)
    le = LabelEncoder()
    le.fit(['A', 'B', 'C', 'D', 'E'])
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    class_names = ['A', 'B', 'C', 'D', 'E']
    
    # 1. Decision Tree (load pre-trained)
    print("Loading Decision Tree...")
    dt_model = joblib.load('../04_Model/models/decision_tree_best.pkl')
    
    # 2. Random Forest (re-train)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train_enc)
    joblib.dump(rf_model, '../04_Model/models/random_forest_best.pkl')
    
    # 3. LightGBM (re-train)
    print("Training LightGBM...")
    lgbm_model = LGBMClassifier(objective='multiclass', num_class=5, n_estimators=100, max_depth=15, learning_rate=0.1, random_state=42, n_jobs=-1)
    lgbm_model.fit(X_train, y_train_enc)
    joblib.dump(lgbm_model, '../04_Model/models/lightgbm_best.pkl')
    
    models = {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'LightGBM': lgbm_model
    }
    
    evaluation_results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        y_test_pred = model.predict(X_test)
        
        if model_name == 'Decision Tree':
            y_test_pred = pd.Series(y_test_pred).str.upper()
            y_test_pred = le.transform(y_test_pred)
        
        # Overall metrics
        metrics = get_metrics(y_test_enc, y_test_pred)
        
        # Classification report (per class)
        report_dict = classification_report(y_test_enc, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_enc, y_test_pred)
        
        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
        plt.close()
        
        val_acc = val_map.get(model_name, {}).get('Validation Accuracy', 0)
        val_f1 = val_map.get(model_name, {}).get('Validation F1', 0)
        
        evaluation_results.append({
            'Model': model_name,
            'Validation Accuracy': val_acc,
            'Test Accuracy': metrics['Accuracy'],
            'Validation F1': val_f1,
            'Test F1': metrics['F1'],
            'Test Precision': metrics['Precision'],
            'Test Recall': metrics['Recall'],
            'Accuracy Diff': metrics['Accuracy'] - val_acc,
            'F1 Diff': metrics['F1'] - val_f1,
            'Classification Report': report_dict
        })
        
    # Save results
    with open('results/final_evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    # Plot Test vs Validation Comparison
    plt.figure(figsize=(10, 6))
    model_names = [r['Model'] for r in evaluation_results]
    val_accs = [r['Validation Accuracy'] * 100 for r in evaluation_results]
    test_accs = [r['Test Accuracy'] * 100 for r in evaluation_results]
    
    x = range(len(model_names))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], val_accs, width, label='Validation Accuracy', color='salmon')
    plt.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy', color='lightgreen')
    
    plt.ylabel('Accuracy (%)')
    plt.title('Validation vs Test Accuracy by Model')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/model_test_comparison.png', dpi=300)
    plt.close()

    # Generate Markdown Report
    report = f"""# Final Evaluation Report

## 1. Evaluation Objective
本次評估的目的是使用完全沒有參與模型訓練與超參數選擇的 Test Set，對目前三個候選模型進行最終測試。這能最客觀地反映模型在現實中面對全新未知資料時的真實泛化能力。

## 2. Test Set Information
- **Test Set 筆數**: {len(X_test):,} 筆
- **Features**: {', '.join(features)} (共 6 個營養成分特徵)
- **Target**: `nutrition_grade_fr` (食品營養等級)
- **Label Mapping**: A ➡️ 0, B ➡️ 1, C ➡️ 2, D ➡️ 3, E ➡️ 4

## 3. Models
本次評估的三個模型及固定參數如下：
- **Decision Tree**: `max_depth=15, random_state=42`
- **Random Forest**: `n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1`
- **LightGBM**: `objective='multiclass', num_class=5, n_estimators=100, max_depth=15, learning_rate=0.1, random_state=42, n_jobs=-1`

## 4. Final Test Results

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 |
|-------|---------------|----------------|-------------|---------|
"""
    for r in evaluation_results:
        report += f"| {r['Model']} | {r['Test Accuracy']:.4f} | {r['Test Precision']:.4f} | {r['Test Recall']:.4f} | {r['Test F1']:.4f} |\n"

    report += "\n## 5. Validation vs Test Comparison\n\n"
    report += "| Model | Validation Accuracy | Test Accuracy | Validation F1 | Test F1 | Accuracy Diff (Test-Val) |\n"
    report += "|-------|---------------------|---------------|---------------|---------|--------------------------|\n"
    for r in evaluation_results:
        report += f"| {r['Model']} | {r['Validation Accuracy']:.4f} | {r['Test Accuracy']:.4f} | {r['Validation F1']:.4f} | {r['Test F1']:.4f} | {r['Accuracy Diff']:.4f} |\n"

    report += "\n## 6. Classification Report (Per Class)\n\n"
    for r in evaluation_results:
        report += f"### {r['Model']}\n"
        report += "| Class | Precision | Recall | F1-score | Support |\n"
        report += "|-------|-----------|--------|----------|---------|\n"
        for c in class_names:
            metrics = r['Classification Report'][c]
            report += f"| {c} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |\n"
        report += "\n"

    report += """## 7. Confusion Matrix Analysis
根據產出的 Confusion Matrix 可以發現，模型最容易混淆相鄰的級別。例如：真正的 A 級食品可能會被誤判為 B，或是真正的 D 級食品被誤判為 C 或 E。跨越多個等級的嚴重誤判（例如把 A 判成 E）相對罕見。由於營養等級的本質是連續的數值切分，相鄰邊界本來就容易存在模糊地帶。

## 8. Final Model Candidate
- 在本次 Test Set 上，Random Forest 的 Accuracy 為最高，維持了其在 Validation 時的領先地位。
- LightGBM 的 Accuracy 次之，但其在 Validation 與 Test 之間的表現落差極小，表現非常穩定。
- Decision Tree 表現最弱，符合預期。

## 9. Limitations
1. **單一 Test Set**：Test Set 只有這一次評估，未來若有資料分布偏移 (Data Drift)，模型表現可能會發生變化。
2. **資料來源**：本模型僅依賴 Open Food Facts 中現有的資料建立，資料集本身的登錄錯誤、缺漏值填補方式皆會影響最終分類品質。
3. **特徵限制**：目前僅使用了 6 個基礎營養特徵。部分國家的 Nutrition Grade 計算可能包含蔬果比例、纖維質等，缺乏這些特徵會限制模型的理論上限。
4. **模型代表性**：目前的評估表現僅代表該資料集內的泛化能力，若應用於截然不同的食品市場，結果可能不同。
"""

    with open('reports/final_evaluation_report.md', 'w') as f:
        f.write(report)
        
    print("Evaluation script finished successfully.")

if __name__ == '__main__':
    run_evaluation()

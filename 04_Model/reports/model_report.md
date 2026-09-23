# Decision Tree Baseline Model Report

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
- **Train**: 124,518 筆
- **Validation**: 26,683 筆

## 5. Decision Tree 基本設定
- `random_state`: 42
- `max_depth`: None (樹會完全生長直到每個葉節點都是純的)
- `criterion`: gini

## 6. Train 評估結果
- **Accuracy**: 0.9937
- **Precision (weighted)**: 0.9937
- **Recall (weighted)**: 0.9937
- **F1-score (weighted)**: 0.9937

## 7. Validation 評估結果
- **Accuracy**: 0.6860
- **Precision (weighted)**: 0.6855
- **Recall (weighted)**: 0.6860
- **F1-score (weighted)**: 0.6857

## 8. Train vs Validation 差異
- **Accuracy 差異**: 30.77%
Train 的準確率幾乎達到 100%，但 Validation 的準確率卻掉到 0.6860。

## 9. 是否有明顯 Overfitting
**是，存在非常嚴重的 Overfitting（過度擬合）。**
因為沒有設定 `max_depth`，Decision Tree 無限制地向下生長，將 Train data 中的雜訊與特例都死背下來（Memorization），導致 Train Accuracy 高達近乎完美的 1.0，但面對從未見過的 Validation data 時，泛化能力 (Generalization) 表現大幅衰退。

## 10. 下一步建議
1. **設定 max_depth (Pruning)**：限制樹的最大深度，迫使模型學習更通用的規則，而不是死背每一筆資料。
2. **進行超參數搜尋 (Hyperparameter Tuning)**：使用 Grid Search 或 Random Search 找出最佳的 `max_depth`、`min_samples_split` 等參數。
3. 等參數調整完成並解決 Overfitting 後，再使用 Test dataset 進行最終評估。

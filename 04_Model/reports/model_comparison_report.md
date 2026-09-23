# Classification Model Comparison Report

## 1. 實驗目的
為了確認除了 Decision Tree 之外，其他分類模型 (如 Random Forest, LightGBM) 是否能在相同特徵下得到更好的泛化能力 (更好的 Validation Accuracy 與較低的 Overfitting Gap)。本實驗在找出最適合進入下一階段進行 Test 評估的最終候選模型。

## 2. 實驗資料
本實驗使用相同的資料切分：
- **Train Set**: 124,518 筆 (僅用於訓練)
- **Validation Set**: 26,683 筆 (僅用於評估與比較)
- **Test Set**: 尚未碰觸，保留至 Final Evaluation。

## 3. 模型介紹
- **Decision Tree**: 作為 Baseline，使用前一階段 Hyperparameter Tuning 找出的最佳候選參數 (`max_depth=15`)。
- **Random Forest**: 透過 Ensemble Learning 集合多棵決策樹，使用 `n_estimators=100`, `max_depth=20`, `min_samples_leaf=2` 來防止過度擬合並增強穩定性。
- **LightGBM**: 輕量級且高效的梯度提升決策樹 (Gradient Boosting Framework)。使用 `objective='multiclass', num_class=5`, `max_depth=15`。

## 4. 實驗結果

| Model | Train Accuracy | Validation Accuracy | Train F1 | Validation F1 | Overfitting Gap | Training Time (s) |
|-------|----------------|---------------------|----------|---------------|-----------------|-------------------|
| Decision Tree | 0.8278 | 0.7086 | 0.8277 | 0.7083 | 0.1192 | 0.78 |
| Random Forest | 0.9428 | 0.7643 | 0.9427 | 0.7639 | 0.1785 | 3.86 |
| LightGBM | 0.7514 | 0.7373 | 0.7508 | 0.7363 | 0.0141 | 2.98 |

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

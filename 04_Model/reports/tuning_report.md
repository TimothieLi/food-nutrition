# Decision Tree Hyperparameter Tuning Report

## 1. 為什麼要進行 Hyperparameter Tuning
目前專案中建立的 Baseline Decision Tree 因為沒有限制決策樹的生長（`max_depth=None`），導致模型過度擬合（Overfitting）訓練資料，雖然在 Train Set 上表現完美，但在 Validation Set 上泛化能力不佳。因此，我們透過限制樹的深度來調整模型複雜度，期望能降低 Overfitting 並提升在未知資料上的表現。

## 2. Baseline Decision Tree 的問題
Baseline 模型在 Train 上的 Accuracy 接近 99.4%，但在 Validation 上僅約 68.6%，這兩者之間有超過 30% 的極大落差。這種「死背資料」的情況意味著模型記住了訓練集裡的雜訊，而無法找出具有一般性的規則。

## 3. 測試了哪些參數
為了控制模型的複雜度，本次測試調整了 `max_depth`（最大深度）這個參數，並測試了以下數值：
- 3
- 5
- 8
- 10
- 15
- None (Baseline 對照組)

其餘參數皆維持 `DecisionTreeClassifier` 預設值（例如 `criterion='gini'`, `min_samples_split=2`），並固定 `random_state=42`。

## 4. 各組 Validation 結果與差距比較

| max_depth | Train Accuracy | Validation Accuracy | Train F1 | Validation F1 | Accuracy 差距 (Train - Val) |
|-----------|----------------|---------------------|----------|---------------|-----------------------------|
| 3 | 0.4988 | 0.4982 | 0.4500 | 0.4491 | 0.0006 |
| 5 | 0.5871 | 0.5863 | 0.5810 | 0.5803 | 0.0007 |
| 8 | 0.6641 | 0.6619 | 0.6618 | 0.6592 | 0.0023 |
| 10 | 0.7069 | 0.6880 | 0.7077 | 0.6887 | 0.0189 |
| 15 | 0.8278 | 0.7086 | 0.8277 | 0.7083 | 0.1192 |
| None | 0.9937 | 0.6860 | 0.9937 | 0.6857 | 0.3077 |

## 5. Train / Validation 差距如何變化
從上表可以看出，當 `max_depth` 越小（例如 3 或 5）時，Train 和 Validation 的差距非常小，模型沒有 Overfitting，但整體的 Accuracy 和 F1 也較低（Underfitting）。
隨著 `max_depth` 增加（例如 8 到 15），Validation 的表現會先上升然後開始停滯甚至稍微下降，而 Train 的表現會持續上升，這導致兩者之間的差距不斷擴大。
當 `max_depth` 為 None 時，差距達到最大（超過 30%）。

## 6. 哪一組參數最適合作為目前的候選模型
綜合觀察 Validation Accuracy 與 Validation F1，表現最好的模型是 **`max_depth` = 15** 的這一組。
它在 Validation 上取得了最佳的分數，並且相對於 Baseline，在保留一定預測能力的同時有效控制了複雜度。

## 7. 是否仍存在 Overfitting
雖然最佳候選模型的 Validation 表現較好，但我們仍可以觀察到 Train 和 Validation 之間有一定程度的差距（約 11.92%）。
這表示相對於完美狀況，仍然存在**輕微到中等程度的 Overfitting**，但已經比 Baseline (`max_depth=None`) 改善非常多，不再是極度嚴重的死背狀況。

## 8. 下一步建議
1. **保留目前的最佳 Decision Tree 模型** 作為目前的基準，等待進入 `05_Evaluation` 使用 Test Set 評估。
2. **嘗試其他演算法**：Decision Tree 本身就容易 Overfitting。可以考慮引入 Random Forest 或 LightGBM 等 Ensemble 演算法，這些演算法天生對抗 Overfitting 的能力更好，有機會將 Validation 表現進一步推高。
3. **特徵工程優化**：如果其他模型也遇到瓶頸，可以回頭考慮是否需要新增衍生特徵或進行不同的轉換。

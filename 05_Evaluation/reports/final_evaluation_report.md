# Final Evaluation Report

## 1. Evaluation Objective
本次評估的目的是使用完全沒有參與模型訓練與超參數選擇的 Test Set，對目前三個候選模型進行最終測試。這能最客觀地反映模型在現實中面對全新未知資料時的真實泛化能力。

## 2. Test Set Information
- **Test Set 筆數**: 26,683 筆
- **Features**: energy_100g, proteins_100g, fat_100g, carbohydrates_100g, sugars_100g, salt_100g (共 6 個營養成分特徵)
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
| Decision Tree | 0.7030 | 0.7030 | 0.7030 | 0.7029 |
| Random Forest | 0.7612 | 0.7617 | 0.7612 | 0.7609 |
| LightGBM | 0.7295 | 0.7290 | 0.7295 | 0.7288 |

## 5. Validation vs Test Comparison

| Model | Validation Accuracy | Test Accuracy | Validation F1 | Test F1 | Accuracy Diff (Test-Val) |
|-------|---------------------|---------------|---------------|---------|--------------------------|
| Decision Tree | 0.7086 | 0.7030 | 0.7083 | 0.7029 | -0.0057 |
| Random Forest | 0.7643 | 0.7612 | 0.7639 | 0.7609 | -0.0031 |
| LightGBM | 0.7373 | 0.7295 | 0.7363 | 0.7288 | -0.0078 |

## 6. Classification Report (Per Class)

### Decision Tree
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.7593 | 0.7597 | 0.7595 | 4053 |
| B | 0.6106 | 0.5907 | 0.6005 | 3931 |
| C | 0.6408 | 0.6506 | 0.6456 | 5807 |
| D | 0.7178 | 0.7358 | 0.7267 | 7713 |
| E | 0.7770 | 0.7536 | 0.7651 | 5179 |

### Random Forest
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.8230 | 0.7972 | 0.8099 | 4053 |
| B | 0.6871 | 0.6390 | 0.6622 | 3931 |
| C | 0.7052 | 0.7202 | 0.7126 | 5807 |
| D | 0.7603 | 0.8116 | 0.7851 | 7713 |
| E | 0.8357 | 0.7967 | 0.8157 | 5179 |

### LightGBM
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| A | 0.8001 | 0.7802 | 0.7900 | 4053 |
| B | 0.6360 | 0.6019 | 0.6185 | 3931 |
| C | 0.6791 | 0.6566 | 0.6677 | 5807 |
| D | 0.7280 | 0.7852 | 0.7555 | 7713 |
| E | 0.8015 | 0.7857 | 0.7935 | 5179 |

## 7. Confusion Matrix Analysis
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

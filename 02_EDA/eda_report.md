# Open Food Facts - Exploratory Data Analysis (EDA) Report

本報告總結了對 Kaggle Open Food Facts 資料集營養特徵與潛在分類標籤的探索與統計分析結果，為建立機器學習（Classification）模型提供依據。

---

## 1. 營養欄位統計與分布 (Nutrition Features)

我們分析了 7 個主要的每 100g 營養特徵。由於部分資料存在明顯的異常值（如大於一億的數值），統計分布圖以合理範圍（營養素 0-100g，熱量 0-4000 kJ）進行了過濾。

| 欄位名稱 | 平均值 (Mean) | 中位數 (Median) | 最小值 (Min) | 最大值 (Max) | 缺失值比例 | 極端異常值數量 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `energy_100g` | 1,125.45 | 1092.00 | 0.00 | 231,199.00 | 17.04% | 113 |
| `proteins_100g` | 53,265.98 | 4.88 | -800.00 | 1.56E+10 | 17.38% | 8 |
| `fat_100g` | 56,065.87 | 5.29 | 0.00 | 1.56E+10 | 21.50% | 4 |
| `carbohydrates_100g` | 56,140.20 | 20.00 | 0.00 | 1.56E+10 | 21.57% | 21 |
| `sugars_100g` | 15.67 | 5.40 | -17.86 | 3520.00 | 21.58% | 20 |
| `fiber_100g` | 384,346.73 | 1.50 | -6.70 | 8.48E+10 | 38.02% | 14 |
| `salt_100g` | 1.94 | 0.55 | 0.00 | 64,312.80 | 18.62% | 164 |

> [!WARNING]
> **資料清理必要性**：從平均值與最大值可以明顯看出，部分特徵（如 `proteins_100g`、`fat_100g`）由於極少數的極端錯誤輸入（高達百億），導致平均值被嚴重扭曲。在建模前必須將數值限制在合理的物理範圍內（例如 `0 <= 數值 <= 100`）。

### 特徵分布圖展示
````carousel
![Energy Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_energy_100g.png)
<!-- slide -->
![Proteins Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_proteins_100g.png)
<!-- slide -->
![Fat Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_fat_100g.png)
<!-- slide -->
![Carbohydrates Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_carbohydrates_100g.png)
<!-- slide -->
![Sugars Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_sugars_100g.png)
<!-- slide -->
![Salt Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_salt_100g.png)
````

---

## 2. 特徵相關性分析 (Correlation Matrix)

![Correlation Heatmap](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/correlation_heatmap.png)

**重點觀察**：
1. **熱量與脂肪高度相關** (`energy` vs `fat`: **0.78**)：這符合營養學常識，因為每克脂肪能提供 9 大卡的熱量，是所有巨量營養素中最高的。
2. **碳水與糖分高度相關** (`carbohydrates` vs `sugars`: **0.67**)：糖分是碳水化合物的一種，兩者自然呈現正相關。
3. **其他特徵間相關性較低**：例如蛋白質與鹽分 (`proteins` vs `salt`: 0.00)，這代表多數特徵提供了獨立的資訊，適合放入模型訓練。

---

## 3. 分類標籤 (Classification Labels) 比較

我們針對兩個潛在的分類任務標籤進行了詳細比較。

### A. `nutrition_grade_fr` (Nutri-Score)
*   **類別數量**：5 個 (A, B, C, D, E)
*   **缺失值**：28.4%
*   **類別平衡**：非常好。各佔比介於 11% ~ 20% 之間，沒有任何一類過度稀少。
    *   d: 20.3%
    *   c: 14.8%
    *   e: 14.1%
    *   a: 11.3%
    *   b: 10.9%

![Nutrition Grade Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_nutrition_grade_fr.png)

### B. `pnns_groups_1` (食品大分類)
*   **類別數量**：14 個
*   **缺失值**：63.8% (包含 12.2% 標記為 unknown)
*   **類別平衡**：極度不平衡。
    *   `unknown` 佔據了絕大部分 (12.2% 總資料)。
    *   接下來最大的類別 `Sugary snacks` 僅佔 4.1%。
    *   最小的類別如 `salty-snacks` 甚至只有 1 筆。

![PNNS Groups Distribution](/Users/timothy/.gemini/antigravity-ide/brain/d11950b5-4a1d-4104-90b0-962c849952a4/dist_pnns_groups_1.png)

### 任務比較與評估

| 評估項目 | 任務 A (`nutrition_grade_fr`) | 任務 B (`pnns_groups_1`) |
| :--- | :--- | :--- |
| **Label 品質** | 高 (官方計算的客觀標準) | 差 (分類模糊、大小寫不統一、太多 unknown) |
| **資料保留量 (非 NaN)** | 約 250,000 筆 | 約 129,000 筆 |
| **類別數量** | 5 類 (適合初學 Demo) | 14 類 |
| **類別平衡** | **非常平衡** | **嚴重不平衡** |
| **模型解釋性** | **極高** (Nutri-score 本就是依營養成分扣分/加分) | 中低 (營養素難以完美反推食品種類) |
| **適合 Decision Tree** | **非常適合** | 樹的深度與分支會非常混亂 |

---

## 4. 總結與建議 (Recommendations)

基於上述 EDA 結果，強烈建議採用以下方案進行期末 Demo：

> [!IMPORTANT]
> **目前最推薦的 Classification 任務**
> 使用機器學習預測食品的 **`nutrition_grade_fr` (A~E 健康星級)**。

> [!TIP]
> **建議使用的 Features**
> `energy_100g`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`
> *(註：捨棄 `fiber_100g`，因其缺失值高達 38%，保留它會損失過多乾淨的樣本)*

> [!CAUTION]
> **需要清理的資料**
> 1. **缺失值刪除**：將這 6 個 Feature 及 Label 欄位中有 `NaN` 的資料整筆刪除。
> 2. **極端值過濾 (Outlier Filtering)**：
>    - 所有重量成分 (`proteins`, `fat`, `carbohydrates`, `sugars`, `salt`) 必須在 **0 到 100 之間**。
>    - 熱量 (`energy_100g`) 必須在 **0 到 4000 之間**。
>    - 不符合上述條件的 Row 應被刪除。

### 下一步 Data Preprocessing 應該做什麼？
1. **載入資料**並只提取上述推薦的 Feature 與 Label 欄位。
2. 執行 **Drop NA** 與 **Outlier 條件過濾**。
3. 為了課堂 Demo 的圖表美觀與執行效率，從清理完的資料集中 **隨機抽樣 (Sample) 10,000 筆資料**。
4. 將 Label 轉換為模型可讀的格式 (如有必要)。
5. 使用 `train_test_split` 切分 80% 訓練集與 20% 測試集。
6. 將清理好的 DataFrame 另存為新的精簡版 CSV (`cleaned_food_data.csv`)，作為下一步訓練模型的輸入。

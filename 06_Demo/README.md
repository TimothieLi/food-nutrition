# AI 食品營養標示分析器 Demo

這是一個整合 **Google Gemini Vision API** 與 **Random Forest 分類模型** 的展示專案。使用者可以拍攝或上傳台灣／國際食品的營養標示，系統透過大語言模型自動擷取營養數值，進行完整的資料驗證與例外處理，再送入機器學習模型預測 Nutrition Grade（A～E），並由 Gemini 生成結構化的 AI 營養觀察報告。

---

## 1. 如何啟動 Demo

### 步驟 1：設定 API Key
本專案需要 Google Gemini API Key 以驅動 OCR 辨識與 AI 觀察報告。請在 `06_Demo` 目錄下建立 `.env` 檔案：

```env
GEMINI_API_KEY=您的_API_KEY
```

> 可前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請。

### 步驟 2：安裝 Python 套件

進入 `06_Demo` 目錄後執行：

```bash
pip install -r requirements.txt
```

### 步驟 3：啟動 Web App

```bash
streamlit run app.py
```

程式將自動開啟瀏覽器並顯示 Web UI。

---

## 2. 主要功能

### 📸 OCR 辨識與手動輸入
- 支援相機拍攝或圖片上傳，呼叫 **Gemini Vision API**（`gemini-2.5-flash`）自動識別營養成分並完成單位轉換（kcal、公克、毫克）。
- 若圖片無法辨識，系統允許使用者全手動輸入數值。

### ✅ 輸入資料驗證與例外處理（`utils/validator.py`）
每次送出分析前，系統會自動執行以下驗證，確保資料品質：

| 驗證規則 | 說明 |
|---|---|
| 全零值偵測 | 六項數值全為 0 時攔截，提示使用者 |
| 負數檢查 | 拒絕任何負數營養成分輸入 |
| 熱量上限 | 超過 1000 kcal/100g 視為不合理 |
| 糖分合理性 | 糖分 > 碳水化合物時標記為異常 |
| 宏量總和 | 蛋白質 + 脂肪 + 碳水 > 105g/100g 時提示 |
| 鈉鹽換算 | 鹽分換算鈉含量過高時警告 |

### 🤖 AI Nutrition Grade 預測
- 使用訓練完成的 Random Forest 分類模型（`random_forest_best.pkl`，Test Accuracy：76.12%）。
- 輸入 6 項特徵：`energy_100g (kJ)`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`。
- 預測結果為 A（最佳）～ E（最差）五個等級。

### 💬 結構化 AI 營養觀察報告（`utils/llm_vision.py`）
AI 觀察報告採用固定 4 段結構輸出，避免單純重複數值，強調實用飲食建議：

1. 🥗 **營養特徵摘要** — 整體特色，不重複原始數字
2. ⚠️ **值得注意的營養項目** — 偏高/偏低的成分及其意義
3. 💡 **飲食建議** — 具體、可操作的建議（如適量、搭配、替代品）
4. 📌 **整體評估** — 以一句話總結這項食品的飲食定位

### 🔁 API 錯誤重試與容錯
- 自動對 503 Service Unavailable、429 Rate Limit 等暫時性錯誤執行指數退避重試（最多 3 次）。
- 依錯誤類型分類顯示友善提示訊息，並避免在 log 中洩漏 API Key。

---

## 3. Demo 操作流程

```
1. 上傳 / 拍攝食品營養標示圖片
        ↓
2. Gemini Vision API 自動辨識並填入數值（或手動輸入）
        ↓
3. 使用者確認 / 修改數值
        ↓
4. 資料驗證（全零、負數、合理性等）
        ↓
5. Random Forest 模型預測 Nutrition Grade（A～E）
   + Gemini 生成 4 段結構化 AI 營養觀察
        ↓
6. 同步顯示分級結果與完整 AI 觀察報告
```

---

## 4. 專案架構

```
06_Demo/
├── app.py                  # Streamlit 主程式（UI 流程控制）
├── requirements.txt        # Python 相依套件
├── .env                    # API Key（請自行建立，勿提交）
├── utils/
│   ├── validator.py        # 輸入資料驗證與例外處理
│   ├── model_runner.py     # Random Forest 模型載入與推論
│   └── llm_vision.py       # Gemini OCR + 4段式 AI 觀察報告生成
├── tests/
│   ├── test_validation.py           # 資料驗證單元測試
│   ├── test_streamlit_app.py        # Streamlit AppTest UI 測試
│   ├── test_advice_and_order.py     # 顯示順序與觀察報告整合測試
│   └── test_nutrition_advice_quality.py  # AI 觀察品質測試
└── assets/                 # 靜態資源（示意圖等）
```

---

## 5. Random Forest 模型說明

| 項目 | 內容 |
|---|---|
| 模型路徑 | `04_Model/models/random_forest_best.pkl` |
| 輸入特徵 | `energy_100g (kJ)`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g` |
| 輸出 | Nutrition Grade：A / B / C / D / E |
| Test Accuracy | 76.12% |

> **UI 單位解耦**：介面顯示大卡（kcal），送入模型前自動轉換為千焦耳（kJ，× 4.184）。

---

## 6. 執行自動化測試

```bash
cd 06_Demo
pytest tests/ -v
```

目前共 31 項測試，涵蓋資料驗證、模型推論、顯示順序、API 錯誤處理與 AI 觀察品質。

---

> ⚠️ **聲明**：本模型預測與 AI 觀察結果不代表完整的健康評估。AI 模型僅根據統計資料學習到的相關性進行預測，不可作為醫療或健康診斷之依據。

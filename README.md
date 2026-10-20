# AI 食品分類與營養標示分析器 (Food Nutrition Grade Predictor & AI Analyzer)

本專案是一個整合 **機器學習 (Machine Learning)** 與 **生成式大語言模型 (Vision LLM)** 的食品營養標示分析系統。以 [Open Food Facts](https://world.openfoodfacts.org/) 國際食品資料集為基礎，透過隨機森林 (Random Forest) 模型預測食品營養等級 (Nutrition Grade: A～E)，並結合 Google Gemini Vision API 自動辨識營養成分標示，提供深入、專業且具備行動建議的「四段式 AI 營養觀察報告」。

---

## 專案架構

本專案依照資料科學與機器學習工程標準流程劃分為六個核心階段：

* **`01_Data/`**：原始資料與清洗後的資料存放區 (大型檔案已透過 `.gitignore` 排除管理)。
* **`02_EDA/`**：探索性資料分析 (EDA) 腳本與數值分佈視覺化圖表。
* **`03_Preprocessing/`**：資料清洗、異常值過濾（0 <= 營養素 <= 100g）與 Train / Validation / Test 資料集切分。
* **`04_Model/`**：Decision Tree、Random Forest 與 LightGBM 等模型的訓練、超參數調整與效能比較。
* **`05_Evaluation/`**：使用獨立 Test Set 進行最終模型評估，產生 Confusion Matrix 與 Classification Report。
* **`06_Demo/`**：基於 Streamlit 建立的互動式 Web 應用程式，整合 Gemini Vision 辨識、嚴謹的輸入資料驗證、隨機森林分級預測與結構化 AI 營養觀察。

---

## 核心成果與特色

1. **高準確率模型**：
   - 最終選用之 Random Forest 分類模型在獨立 Test Set 達到 **76.12% Accuracy**，各級別 (A～E) 具備均衡穩健的泛化能力。
   - 特徵維度：熱量 (`energy_100g` [kJ])、蛋白質、脂肪、碳水化合物、糖、食鹽。
2. **多模態 Vision LLM 自動解析**：
   - 支援手機相機拍照與圖片上傳，透過 Gemini Vision 自動讀取中英文營養標示並辨識計算基準（每 100g、每 100ml、每份或整包）。
3. **嚴謹的資料驗證與例外防呆機制**：
   - 徹底拒絕將未取得之缺失值填為 0。
   - 防範負數、NaN/Inf 等無效格式。
   - 物理與生理合理性檢查（糖不可大於碳水化合物、單一營養素不可超過 100g、熱量物理極限檢查）。
   - 全零特殊食品（如純水、無糖茶）確認防護機制。
4. **高彈性重試與優雅降級架構**：
   - 具備 503 UNAVAILABLE 與網路逾時之指數退避（Exponential Backoff）重試機制。
   - 狀態管理（Session State）防重複呼叫設計，避免 Streamlit 重新渲染浪費 API 額度。
5. **實用四段式 AI 營養觀察報告**：
   - **🥗 營養特徵摘要**：客觀總結食品主要能量與巨量營養組成。
   - **⚠️ 值得注意的營養項目**：針對最關鍵的 1～3 項營養素深入解讀（如高糖、高鈉提醒）。
   - **💡 飲食建議**：提供 1～3 項具體、可執行的飲食挑選與搭配建議。
   - **📌 整體評估**：中立說明模型預測分級，提醒考量個人食用份量與整天飲食均衡。

---

## 快速啟動

### 1. 環境安裝
建議使用 Python 3.10+ 環境：

```bash
# 安裝所有相依套件
pip install -r requirements.txt
```

### 2. 設定 Gemini API Key
前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請 API Key，並在 `06_Demo/.env` 中設定：

```env
GEMINI_API_KEY=您的_API_KEY
```

### 3. 啟動 Web App
```bash
cd 06_Demo
streamlit run app.py
```
啟動後於瀏覽器開啟 `http://localhost:8501` 即可開始體驗！

---

## 執行自動化測試

本專案在 `06_Demo/tests/` 包含涵蓋資料驗證、流程控制、API 錯誤處理與觀察品質的完整自動化測試（共 31 項測試）：

```bash
cd 06_Demo
python -m unittest discover tests -v
```

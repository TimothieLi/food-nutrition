# AI 食品營養標示分析器 Demo

這是一個整合最新的 **Google Gemini Vision API** 與 **Random Forest 分類模型** 的展示專案。本專案允許使用者拍攝台灣或國際食品的營養標示，透過大語言模型自動擷取營養數值，隨後預測該食品的 Nutrition Grade (A～E)，並產生個人化的營養觀察建議。

## 1. 如何啟動 Demo

### 步驟 1：設定 API Key
由於本專案依賴 Google Gemini 模型進行圖像辨識與建議生成，請先在 `06_Demo` 目錄下建立 `.env` 檔案，並填入您的 API Key：
```env
GEMINI_API_KEY=您的_API_KEY
```
*(您可以前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請)*

### 步驟 2：安裝 Python 套件
進入 `06_Demo` 目錄後，安裝所需的相依套件：
```bash
pip install -r requirements.txt
```

### 步驟 3：啟動 Web App
```bash
streamlit run app.py
```
程式將會自動開啟瀏覽器並顯示 Web UI。

## 2. 使用哪些主要套件
- **Frontend / Web UI**: `streamlit` (支援相機與檔案上傳，適合互動展示)
- **Vision & LLM Engine**: `google-genai` (負責辨識營養標示圖片，以及生成營養觀察報告)
- **Machine Learning**: `scikit-learn` (載入既有隨機森林模型), `pandas` (資料處理), `joblib` (反序列化模型)

## 3. Demo 操作流程
1. **上傳/拍攝**: 在首頁使用相機拍攝食品營養標示，或是上傳圖片檔。
2. **AI 視覺辨識**: 背景呼叫 Gemini Vision API (gemini-2.5-flash) 進行文字圖像分析與單位轉換（如 kcal、公克）。
3. **確認/修改**: 擷取到的營養數值將自動填入中文表單。使用者可檢查並手動修改數值。
4. **模型預測**: 按下「確認」按鈕後，程式會將 UI 的熱量 (kcal) 轉換為模型需要的 (kJ)，並送入機器學習模型。
5. **檢視結果**: 畫面將會顯示預測的 Nutrition Grade (A~E)，以及一段由 AI 專門為該數值生成的「營養觀察與建議」。

## 4. Random Forest 的模型來源
本專案所使用的機器學習模型保留了 `04_Model` 階段的實驗產出，未做任何更動：
- **路徑**: `04_Model/models/random_forest_best.pkl`
- **輸入特徵**: 6 項數值 (`energy_100g` [kJ], `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`)
- **Test Accuracy**: 76.12%

## 5. 系統防呆與設計細節
- **單位解耦**: 使用者介面 (UI) 顯示最直觀的「大卡 (kcal)」，而底層傳給預測模型的資料流則會自動轉換回模型訓練時使用的「千焦耳 (kJ)」。
- **動態生成報告**: 利用 Gemini 將生硬的數值轉化為一般大眾容易理解的白話文觀察報告。
- **手動 Fallback**: 如果拍攝的角度過於模糊導致無法辨識，系統依然允許使用者全手動輸入數值來體驗模型預測。

---
> ⚠️ **聲明**：本模型預測與 AI 觀察結果不代表完整的健康評估。AI 模型僅是根據大量訓練資料中的特徵與標籤學習到統計相關性，不可作為醫療與健康診斷之依據。

# AI 食品營養標示分析器 Demo

這是一個針對食品暨應用生物科技相關課程期末展示所設計的機器學習展示專案。本專案整合了 OCR (文字辨識) 技術與 Random Forest 分類模型，允許使用者拍攝食品營養標示，擷取營養數值後，預測該食品的 Nutrition Grade (A～E)。

## 1. 如何啟動 Demo

本 Demo 依賴 `pytesseract` 作為 OCR 引擎，以及 `streamlit` 作為 Web UI。

### 步驟 1：安裝 Tesseract 系統套件 (Mac)
由於 OCR 依賴系統底層引擎，請先在 Terminal 執行：
```bash
brew install tesseract
```

### 步驟 2：安裝 Python 套件
進入 `06_Demo` 目錄後執行：
```bash
pip install -r requirements.txt
```

### 步驟 3：啟動 Web App
```bash
streamlit run app.py
```
程式將會自動開啟瀏覽器並顯示 Web UI。

## 2. 使用哪些套件
- **Frontend / Web UI**: `streamlit` (輕量化、支援相機與檔案上傳，適合機器學習 Demo)
- **OCR Engine**: `pytesseract` (Tesseract OCR 的 Python wrapper), `Pillow` (影像處理)
- **Machine Learning**: `scikit-learn` (載入模型與前處理), `pandas` (資料表操作), `joblib` (反序列化模型)

## 3. Demo 操作流程
1. **上傳/拍攝**: 在首頁使用相機拍攝食品營養標示，或是直接上傳圖片檔。
2. **OCR 辨識**: 程式會在背景讀取圖片並進行文字分析。
3. **確認/修改**: OCR 擷取到的營養數值將會自動填入表單。使用者可以檢查並手動修改錯誤或遺漏的數字。
4. **模型預測**: 按下「確認並開始 AI 分析」按鈕。
5. **檢視結果**: 畫面將會顯示預測的 Nutrition Grade (A~E)，並帶有醒目的視覺化設計。

## 4. OCR 的限制
- 目前使用 `pytesseract` 搭配 Regex 來尋找特定的英文關鍵字 (例如 Energy, Protein 等)。
- 如果包裝的反光嚴重、字體扭曲、或是非英文/常見格式的標示，OCR 有高機率會遺漏或誤判。
- 這就是為什麼系統設計成**「必須先讓使用者確認後再預測」**，並且隨時提供**「完全手動輸入模式」**作為 Fallback。

## 5. Random Forest 的模型來源
本專案所使用的 ML 模型來自於 `04_Model` 階段的實驗產出：
- **路徑**: `04_Model/models/random_forest_best.pkl`
- **輸入特徵**: 6 項數值型營養成分 (`energy_100g`, `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`)
- **Test Accuracy**: 76.12%
- **重要聲明**: 所有的特徵前處理皆與模型訓練時完全一致，並未重新進行訓練或調整。

## 6. 已完成的例外處理
- **完全找不到營養資訊**：例如拍攝食品正面。系統會提示警告，並自動啟動「手動輸入表單」，讓 Demo 能順利進行。
- **資訊不完整**：若 OCR 僅找到部分欄位，其餘欄位會顯示預設值 0.0，並提示使用者「請手動補充缺漏的欄位」。
- **型別與防呆設計**：UI 表單限制使用者只能輸入數字 (浮點數)，並避免了負數輸入，確保不會發生 Format Exception 導致程式崩潰。

## 7. 測試結果
以下情境皆已在本地端完成邏輯驗證：
- [x] **Test 1**: 正常上傳照片，成功自動填入數據並預測 A～E。
- [x] **Test 2**: 圖片僅有部分關鍵字時，未辨識成功的欄位為 0.0，使用者補齊後可成功預測。
- [x] **Test 3 & 4**: 拍攝食品正面或一般風景圖，觸發 `any()` 防線，顯示「找不到足夠的營養資訊」，安全降級至手動輸入。
- [x] **Test 5 & 6**: 完全不使用相機/圖片上傳，直接拉到底部更改數值並點擊分析，Random Forest 可正確給予預測等級。

---
> ⚠️ **聲明**：本模型預測結果不代表完整的健康評估。AI 模型僅是根據大量訓練資料中的特徵與標籤學習到統計相關性。

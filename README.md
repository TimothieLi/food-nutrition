# AI 食品分類器專案 (Nutrition Grade Predictor)

這是一個完整的 Machine Learning Classification 專案，使用 Open Food Facts 資料，根據食品的營養資訊（如熱量、蛋白質、脂肪等）來預測該食品的營養等級 (Nutrition Grade: A~E)。

## 專案結構

本專案依照資料科學標準流程分為六個階段：

* **`01_Data/`**: 原始資料與清洗後的資料存放區 (因檔案過大，已加入 `.gitignore`)
* **`02_EDA/`**: 探索性資料分析 (EDA) 腳本與視覺化圖表
* **`03_Preprocessing/`**: 資料前處理與 Train/Validation/Test 拆分
* **`04_Model/`**: Decision Tree, Random Forest, LightGBM 等模型的訓練、超參數調整與比較報告
* **`05_Evaluation/`**: 使用獨立 Test Set 進行最終模型評估，產生 Confusion Matrix 與 Classification Report
* **`06_Demo/`**: 基於 Streamlit 與 OCR (Tesseract) 實作的 Web UI，提供拍照/上傳營養標示自動辨識並預測等級的功能

## 如何執行 Demo

請進入 `06_Demo` 資料夾，並參考該目錄下的 `README.md` 以取得完整的安裝與啟動指示：

```bash
cd 06_Demo
pip install -r requirements.txt
streamlit run app.py
```
*(請注意：執行 Demo 的 OCR 功能需要系統預先安裝 Tesseract)*

## 模型成果
本專案最終評估選用的 Random Forest 模型，在未看過的 Test Set 上達到了 **76.12%** 的 Accuracy，並具備穩定且良好的泛化能力 (Train/Validation gap 縮小至合理範圍)。

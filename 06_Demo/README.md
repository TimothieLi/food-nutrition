# AI 食品營養標示分析器 Demo

這是一個整合 **Google Gemini Vision API** 與 **Random Forest 分類模型** 的端對端營養分析應用程式。使用者可透過拍照或上傳食品營養標示圖片，由 Vision LLM 自動擷取營養數值，經嚴格的資料品質驗證後，由隨機森林模型預測 Nutrition Grade (A～E)，並產生具備行動指引的「四段式 AI 營養觀察報告」。

---

## 1. 如何啟動 Demo

### 步驟 1：設定 API Key
本專案使用 Google Gemini API 進行圖像辨識與營養觀察生成。請在 `06_Demo` 目錄下建立 `.env` 檔案，填入您的 API Key：
```env
GEMINI_API_KEY=您的_API_KEY
```
*(可前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 免費申請)*

### 步驟 2：安裝 Python 套件
進入 `06_Demo` 目錄後安裝相依套件：
```bash
pip install -r requirements.txt
```

### 步驟 3：啟動 Web App
```bash
streamlit run app.py
```
程式將啟動本地伺服器並自動在瀏覽器中開啟 Web UI（預設為 `http://localhost:8501`）。

---

## 2. 系統架構與功能亮點

### 1. 多模態視覺辨識 (Vision OCR)
- 呼叫 `gemini-2.5-flash` 進行食品營養標示圖片分析。
- 自動識別標示計算基準（每 100g、每 100ml、每份或整包）。
- 嚴格遵守「缺失不猜測」原則，未辨識出之欄位維持 `None`，不任意預設填 0。

### 2. 嚴格輸入驗證與防呆 (Validation & Exception Handling)
- 位於 `utils/validator.py`，所有數值必須在進入模型前通過嚴密檢驗：
  - **缺失值檢查**：明確指明缺漏欄位並阻止預測。
  - **負數與格式檢查**：攔截負數、文字格式錯誤與 NaN/Inf。
  - **生物/物理合理性檢查**：糖不可大於碳水化合物；單一營養素在 100g 食品中不得超過 100g；100g 食品熱量不得超過 1000 kcal（物理極限）。
  - **標示基準確認**：非 100g/ml 基準時主動警示需換算。
  - **全零食品確認機制**：六項數值皆為 0 時提示確認，防止 OCR 失誤被誤判為 A 級；若確認為純水或無糖茶，則進入專屬分析流程。

### 3. 結果顯示順序整合與狀態防重複 (State Management)
- **結果同步顯示**：AI 預測分級與營養觀察於背景完成後，在同一結果區塊一次性完整呈現，杜絕分級提前單獨曝光。
- **防止重複呼叫**：API 請求僅於表單送出（`if submitted:`）時觸發一次並存入 `st.session_state`，頁面重新渲染（Rerun）不再重複發送請求，極大節省 API 配額。
- **清除殘留結果**：表單重新提交或修改數值時，立即清除舊的預測與觀察結果。

### 4. 高彈性重試與優雅降級 (Resilience & Retries)
- 面對伺服器高負載（503 UNAVAILABLE）或連線逾時，自動執行最多 3 次之指數退避（Exponential Backoff）重試。
- 精準區分 503 服務忙碌、429 配額超額與 401/403 設定權限問題，不向一般使用者暴露生硬的例外堆疊，遮蔽敏感金鑰。
- 若 LLM 服務持續忙碌，隨機森林分級仍正常呈現，並提供「重新嘗試產生營養觀察」按鈕供單獨重試。

### 5. 實用四段式 AI 營養觀察 (Actionable Nutrition Advice)
改善舊版重複念數字的問題，改以四個明確段落呈現：
- **🥗 營養特徵摘要**：客觀總結能量密度與主要營養成分。
- **⚠️ 值得注意的營養項目**：針對最關鍵的 1～3 項重點深入解讀（如糖分過高、鈉含量顯著）。
- **💡 飲食建議**：提供 1～3 項具體、可執行的飲食挑選、份量換算與食物搭配建議。
- **📌 整體評估**：中立說明模型預測分級，強調綜合膳食考量，不將單一食品貼標籤，無醫療診斷宣稱。

---

## 3. 自動化測試集

本專案在 `tests/` 資料夾內包含完善的單元測試與端對端測試：

```bash
# 執行全部 31 項測試
python -m unittest discover tests -v

# 單獨執行各測試模組
python -m unittest tests/test_validation.py -v              # 資料驗證 9 項測試
python -m unittest tests/test_advice_and_order.py -v        # 順序與 503 重試 8 項測試
python -m unittest tests/test_nutrition_advice_quality.py -v # 觀察品質與情境 7 項測試
python -m unittest tests/test_streamlit_app.py -v          # Streamlit UI 端對端 7 項測試
```

---

## 4. Random Forest 模型規格
- **模型路徑**：`04_Model/models/random_forest_best.pkl`
- **特徵順序**：`energy_100g` (kJ), `proteins_100g`, `fat_100g`, `carbohydrates_100g`, `sugars_100g`, `salt_100g`
- **獨立測試集 Accuracy**：**76.12%**
- **標籤映射**：0=A, 1=B, 2=C, 3=D, 4=E

---
> ⚠️ **聲明**：本系統預測之營養分級與 AI 觀察報告僅供日常飲食參考，非個人健康診斷或醫療處方。食用選擇請依個人體質、活動量及專業醫師/營養師建議評估。

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from google.genai.errors import APIError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(DEMO_DIR)
from dotenv import load_dotenv
load_dotenv()

from utils.llm_vision import generate_nutrition_advice
from utils.validator import validate_nutrition_data
from streamlit.testing.v1 import AppTest

APP_FILE = os.path.join(DEMO_DIR, 'app.py')

class TestNutritionAdviceQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 確保測試執行期間有 API Key，避免 mock 測試因未設定金鑰被前置檢查阻擋
        if not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = "mock_test_key"

    # ------------------------------------------------------------------
    # 測試情境 1：一般營養數據完整的食品
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_01_complete_regular_food(self, mock_generate):
        """測試一般營養數據完整時，AI 營養觀察包含完整的四段結構與中立評估"""
        mock_output = (
            "**🥗 營養特徵摘要**\n"
            "此食品在每 100g 基準下具備適度熱量與良好蛋白質，脂肪與糖含量皆處於適中比例。\n\n"
            "**⚠️ 值得注意的營養項目**\n"
            "• **蛋白質與熱量**：提供穩定的能量補給，適合作為日常正餐中的蛋白質補充搭配。\n\n"
            "**💡 飲食建議**\n"
            "1. **注意份量**：食用前請確認包裝份量換算。\n"
            "2. **多樣搭配**：可搭配高纖蔬果，促進營養均衡。\n\n"
            "**📌 整體評估**\n"
            "根據模型預測結果，本食品被評為 B 級。建議考量整體飲食結構進行搭配，不單依分級決定。"
        )
        mock_resp = MagicMock()
        mock_resp.text = mock_output
        mock_generate.return_value = mock_resp

        data = {
            'energy_100g': 180.0,
            'proteins_100g': 12.0,
            'fat_100g': 5.0,
            'carbohydrates_100g': 20.0,
            'sugars_100g': 3.0,
            'salt_100g': 0.6
        }

        advice, error_info = generate_nutrition_advice(data, grade='B', basis='per_100g')
        self.assertIsNotNone(advice)
        self.assertIsNone(error_info)

        # 驗證四大區塊標題
        self.assertIn("**🥗 營養特徵摘要**", advice)
        self.assertIn("**⚠️ 值得注意的營養項目**", advice)
        self.assertIn("**💡 飲食建議**", advice)
        self.assertIn("**📌 整體評估**", advice)

        # 驗證傳給 Gemini 的 prompt 包含指定品質規範
        sent_prompt = mock_generate.call_args[1]['contents']
        self.assertIn("標示計算基準：每 100 公克 (per 100g)", sent_prompt)
        self.assertIn("180.0 kcal", sent_prompt)
        self.assertIn("機器學習模型預測分級：B 級", sent_prompt)
        self.assertIn("不可只因單一營養素較高，就直接判定食品不健康", sent_prompt)

    # ------------------------------------------------------------------
    # 測試情境 2：糖含量值得注意的食品
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_02_high_sugar_food(self, mock_generate):
        """測試糖含量偏高食品，Prompt 與觀察內容能聚焦於糖與食用份量，且無疾病診斷宣稱"""
        mock_resp = MagicMock()
        mock_resp.text = (
            "**🥗 營養特徵摘要**\n"
            "此食品碳水化合物佔比高，其中糖含量相當顯著。\n\n"
            "**⚠️ 值得注意的營養項目**\n"
            "• **糖含量**：每 100g 含有高達 38g 糖，應特別留意單次食用份量。\n\n"
            "**💡 飲食建議**\n"
            "1. **控制份量**：建議分次食用，避免單次攝取過多糖分。\n"
            "2. **產品比較**：選購同類產品時，可比較每份糖含量較低的替代品。\n\n"
            "**📌 整體評估**\n"
            "根據模型預測結果，本食品被評為 D 級。食用選擇應搭配整日飲食規劃，適量品嚐。"
        )
        mock_generate.return_value = mock_resp

        high_sugar_data = {
            'energy_100g': 420.0,
            'proteins_100g': 3.0,
            'fat_100g': 12.0,
            'carbohydrates_100g': 70.0,
            'sugars_100g': 38.0,
            'salt_100g': 0.3
        }

        advice, error_info = generate_nutrition_advice(high_sugar_data, grade='D', basis='per_100g')
        self.assertIsNotNone(advice)
        self.assertIn("糖含量", advice)
        self.assertIn("**💡 飲食建議**", advice)
        
        # 驗證 prompt 中有強調糖的分析重點
        sent_prompt = mock_generate.call_args[1]['contents']
        self.assertIn("若糖含量較高，提醒使用者注意實際食用份量", sent_prompt)
        self.assertIn("絕無醫療宣稱", sent_prompt)

    # ------------------------------------------------------------------
    # 測試情境 3：鈉/食鹽含量值得注意的食品
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_03_high_sodium_food(self, mock_generate):
        """測試鈉/食鹽偏高食品，聚焦於全天鈉攝取提醒與比較"""
        mock_resp = MagicMock()
        mock_resp.text = (
            "**🥗 營養特徵摘要**\n"
            "本食品風味濃郁，每 100g 之食鹽含量顯著高於一般平均值。\n\n"
            "**⚠️ 值得注意的營養項目**\n"
            "• **食鹽含量**：每 100g 含有 3.2g 食鹽，提醒留意整日飲食中的累積鈉攝取。\n\n"
            "**💡 飲食建議**\n"
            "1. **調配飲食**：當餐若食用此品，其他菜餚可選擇清淡少鹽烹調。\n"
            "2. **增加飲水**：適度補充水分並多攝取含鉀之新鮮蔬果。\n\n"
            "**📌 整體評估**\n"
            "根據模型預測結果，本食品被評為 C 級。請考量全天鈉含量總額，平衡整體飲食。"
        )
        mock_generate.return_value = mock_resp

        high_salt_data = {
            'energy_100g': 260.0,
            'proteins_100g': 9.0,
            'fat_100g': 8.0,
            'carbohydrates_100g': 35.0,
            'sugars_100g': 2.0,
            'salt_100g': 3.2
        }

        advice, error_info = generate_nutrition_advice(high_salt_data, grade='C', basis='per_100g')
        self.assertIsNotNone(advice)
        self.assertIn("食鹽含量", advice)
        self.assertIn("**📌 整體評估**", advice)

        # 驗證 prompt 包含食鹽/鈉重點引導
        sent_prompt = mock_generate.call_args[1]['contents']
        self.assertIn("若食鹽/鈉含量較高，提醒留意全天飲食中的鈉攝取", sent_prompt)

    # ------------------------------------------------------------------
    # 測試情境 4：缺失部分營養數據的食品
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_04_missing_data_food(self, mock_generate):
        """測試當部分數值為 None 時，Prompt 明確標記未提供，絕不偷偷填 0"""
        mock_resp = MagicMock()
        mock_resp.text = (
            "**🥗 營養特徵摘要**\n"
            "本食品提供熱量與碳水化合物資訊，但部分數值未提供完整標示。\n\n"
            "**⚠️ 值得注意的營養項目**\n"
            "• **缺失數值**：蛋白質未提供數據，無法評估其蛋白質供給能力。\n\n"
            "**💡 飲食建議**\n"
            "1. **核對標示**：建議查看實體包裝是否有更完整的成分列表。\n\n"
            "**📌 整體評估**\n"
            "因部分數值未完整提供，分析結果僅供參考，應考量整體膳食。"
        )
        mock_generate.return_value = mock_resp

        missing_data = {
            'energy_100g': 200.0,
            'proteins_100g': None, # 缺失
            'fat_100g': 5.0,
            'carbohydrates_100g': 30.0,
            'sugars_100g': None,   # 缺失
            'salt_100g': 0.2
        }

        advice, error_info = generate_nutrition_advice(missing_data, grade='C', basis='per_100g')
        sent_prompt = mock_generate.call_args[1]['contents']

        # 嚴格驗證：未提供的數值顯示「未提供」，絕對沒有出現 0.0 g 或被當成 0
        self.assertIn("蛋白質：未提供", sent_prompt)
        self.assertIn("糖：未提供", sent_prompt)
        self.assertIn("若有數值為「未提供」，應明確說明目前缺少該項數據無法判斷，絕對不可將缺失或未辨識的數值當作 0", sent_prompt)

    # ------------------------------------------------------------------
    # 測試情境 5：營養數值異常或未通過驗證的食品
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_05_invalid_data_blocked_from_llm(self, mock_generate):
        """測試當數值異常（如負數或糖大於碳水），驗證機制徹底阻擋 LLM 與模型呼叫"""
        invalid_data = {
            'energy_kcal': 300.0,
            'proteins_100g': 5.0,
            'fat_100g': -2.0,  # 異常負數
            'carbohydrates_100g': 10.0,
            'sugars_100g': 25.0,  # 糖 > 碳水
            'salt_100g': 0.5
        }

        val_result = validate_nutrition_data(invalid_data, basis='per_100g')
        self.assertFalse(val_result['is_valid'])
        self.assertEqual(val_result['status'], 'invalid')
        
        # 驗證在此狀態下，絕不呼叫 generate_nutrition_advice
        mock_generate.assert_not_called()

    # ------------------------------------------------------------------
    # 測試情境 6：Gemini API 呼叫失敗時的處理
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_06_api_failure_handling(self, mock_generate):
        """測試 503 等 API 錯誤時，重試後若仍失敗能給予友善提示，且不讓介面崩潰"""
        err_503 = APIError(
            503, 
            {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}
        )
        mock_generate.side_effect = err_503

        data = {
            'energy_100g': 250.0,
            'proteins_100g': 5.0,
            'fat_100g': 5.0,
            'carbohydrates_100g': 40.0,
            'sugars_100g': 10.0,
            'salt_100g': 0.5
        }

        advice, error_info = generate_nutrition_advice(
            data, grade='C', basis='per_100g', max_retries=2, initial_delay=0.01
        )

        self.assertIsNone(advice)
        self.assertIsNotNone(error_info)
        self.assertEqual(error_info['category'], 'transient_service')
        self.assertIn("營養觀察服務目前忙碌中，請稍後重試。", error_info['user_message'])

    # ------------------------------------------------------------------
    # 測試情境 7：全零食品在 UI 呈現四大區塊結構
    # ------------------------------------------------------------------
    def test_07_all_zero_four_sections_ui(self):
        """測試全零食品經使用者確認後，亦能產出一致的四段式結構"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        
        # 填入六項 0
        for i in range(6):
            at.number_input[i].set_value(0.0)
            
        # 勾選全零確認
        at.checkbox[0].check()
        at.button[0].click().run()

        all_text = " ".join([m.value for m in at.markdown])
        self.assertIn("🥗 營養特徵摘要", all_text)
        self.assertIn("⚠️ 值得注意的營養項目", all_text)
        self.assertIn("💡 飲食建議", all_text)
        self.assertIn("📌 整體評估", all_text)
        self.assertIn("Nutrition Grade", all_text)

if __name__ == '__main__':
    unittest.main()

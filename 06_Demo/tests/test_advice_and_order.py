import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from google.genai.errors import APIError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(DEMO_DIR)

from utils.llm_vision import generate_nutrition_advice, classify_api_error
from utils.model_runner import load_model
from streamlit.testing.v1 import AppTest

APP_FILE = os.path.join(DEMO_DIR, 'app.py')

class TestAdviceAndDisplayOrder(unittest.TestCase):

    def setUp(self):
        self.dummy_nutrition = {
            'energy_100g': 350.0,
            'proteins_100g': 8.0,
            'fat_100g': 12.0,
            'carbohydrates_100g': 55.0,
            'sugars_100g': 10.0,
            'salt_100g': 0.8
        }

    # ------------------------------------------------------------------
    # 測試 1：正常輸入，Random Forest 預測成功，營養觀察也成功。
    # 預期：先完成營養觀察，再一起顯示分級、分級說明及營養觀察。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_01_normal_success_unified_display(self, mock_generate):
        mock_resp = MagicMock()
        mock_resp.text = "這是一份均衡且健康的營養觀察建議。"
        mock_generate.return_value = mock_resp

        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        at.number_input[0].set_value(350.0)
        at.number_input[1].set_value(8.0)
        at.number_input[2].set_value(12.0)
        at.number_input[3].set_value(55.0)
        at.number_input[4].set_value(10.0)
        at.number_input[5].set_value(0.8)
        at.button[0].click().run()

        # 驗證統一區塊存在
        all_text = " ".join([m.value for m in at.markdown] + [h.value for h in at.subheader] + [t.value for t in at.title])
        self.assertIn("Nutrition Grade", all_text)
        self.assertIn("AI 食品營養分析結果", all_text)
        
        # 驗證營養觀察與分級說明皆顯示
        self.assertIn("這是一份均衡且健康的營養觀察建議。", all_text)
        self.assertIn("機器學習模型根據每 100g 營養特徵數值之預測結果", " ".join([c.value for c in at.caption] + [m.value for m in at.markdown]))

    # ------------------------------------------------------------------
    # 測試 2：Random Forest 預測成功，但營養觀察 API 回傳 503。
    # 預期：依照有限次數重試。若仍失敗，不崩潰，顯示分級與暫時性錯誤提示。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_02_api_503_retry_and_graceful_handling(self, mock_generate):
        err_503 = APIError(
            503, 
            {'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', 'status': 'UNAVAILABLE'}}
        )
        mock_generate.side_effect = err_503

        # 測試函式本身包含指數退避重試
        advice, error_info = generate_nutrition_advice(
            self.dummy_nutrition, grade='A', max_retries=3, initial_delay=0.01
        )
        self.assertIsNone(advice)
        self.assertIsNotNone(error_info)
        self.assertEqual(error_info['code'], 503)
        self.assertEqual(error_info['category'], 'transient_service')
        self.assertIn("營養觀察服務目前忙碌中，請稍後重試。", error_info['user_message'])
        self.assertEqual(mock_generate.call_count, 3, "應重試 3 次後停止")

        # 測試在 Streamlit UI 表現
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        at.number_input[0].set_value(350.0)
        at.number_input[1].set_value(8.0)
        at.number_input[2].set_value(12.0)
        at.number_input[3].set_value(55.0)
        at.number_input[4].set_value(10.0)
        at.number_input[5].set_value(0.8)
        at.button[0].click().run()

        # 分級正常顯示，不因 503 崩潰
        self.assertTrue(any("Nutrition Grade" in m for m in [m.value for m in at.markdown]))
        # 顯示友善警告訊息
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("營養觀察服務目前忙碌中，請稍後重試" in w for w in warnings))

    # ------------------------------------------------------------------
    # 測試 3：營養觀察第一次呼叫失敗 (503)，第二次重試成功。
    # 預期：成功取得觀察內容，不產生重複或互相矛盾結果。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_03_retry_succeeds_on_second_attempt(self, mock_generate):
        err_503 = APIError(503, {'error': {'code': 503, 'message': 'Temporarily unavailable'}})
        success_resp = MagicMock()
        success_resp.text = "重試後成功產生的營養觀察內容。"

        mock_generate.side_effect = [err_503, success_resp]

        advice, error_info = generate_nutrition_advice(
            self.dummy_nutrition, grade='B', max_retries=3, initial_delay=0.01
        )
        self.assertEqual(advice, "重試後成功產生的營養觀察內容。")
        self.assertIsNone(error_info)
        self.assertEqual(mock_generate.call_count, 2)

    # ------------------------------------------------------------------
    # 測試 4：營養觀察 API 回傳 429 或其他非 503 錯誤。
    # 預期：依照實際錯誤類型處理，不將所有錯誤都顯示為 503。
    # ------------------------------------------------------------------
    def test_04_error_categorization_429_and_403(self):
        err_429 = APIError(429, {'error': {'code': 429, 'message': 'Resource has been exhausted.'}})
        res_429 = classify_api_error(err_429)
        self.assertEqual(res_429['code'], 429)
        self.assertEqual(res_429['category'], 'quota_rate_limit')
        self.assertIn("檢查服務使用額度", res_429['user_message'])

        err_403 = APIError(403, {'error': {'code': 403, 'message': 'The caller does not have permission.'}})
        res_403 = classify_api_error(err_403)
        self.assertEqual(res_403['code'], 403)
        self.assertEqual(res_403['category'], 'config_permission')
        self.assertIn("檢查模型設定或 API 權限", res_403['user_message'])

    # ------------------------------------------------------------------
    # 測試 5：使用者修改營養數值後重新分析。
    # 預期：清除舊結果，使用最新數值重新預測並產生新的營養觀察。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_05_modify_inputs_clears_old_and_updates(self, mock_generate):
        resp1 = MagicMock()
        resp1.text = "【觀察 1】熱量 200 大卡"
        resp2 = MagicMock()
        resp2.text = "【觀察 2】熱量 400 大卡"
        mock_generate.side_effect = [resp1, resp2]

        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        
        # 第一次送出：熱量 200
        at.number_input[0].set_value(200.0)
        at.number_input[1].set_value(5.0)
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(30.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.5)
        at.button[0].click().run()
        self.assertIn("【觀察 1】熱量 200 大卡", " ".join([m.value for m in at.markdown]))

        # 第二次修改熱量為 400 並送出
        at.number_input[0].set_value(400.0)
        at.button[0].click().run()
        
        all_markdown = " ".join([m.value for m in at.markdown])
        self.assertNotIn("【觀察 1】熱量 200 大卡", all_markdown, "舊的觀察文字必須被清除")
        self.assertIn("【觀察 2】熱量 400 大卡", all_markdown, "應顯示最新數值產生的營養觀察")

    # ------------------------------------------------------------------
    # 測試 6：使用者沒有修改資料，只是 Streamlit 重新執行。
    # 預期：不會不必要地重複呼叫營養觀察 API。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_06_rerun_does_not_duplicate_api_calls(self, mock_generate):
        resp = MagicMock()
        resp.text = "分析結果：優質高蛋白。"
        mock_generate.return_value = resp

        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        at.number_input[0].set_value(300.0)
        at.number_input[1].set_value(20.0)
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(40.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.5)
        at.button[0].click().run()

        # 第一次送出，API 呼叫次數為 1
        self.assertEqual(mock_generate.call_count, 1)

        # 模擬 Streamlit 頁面因其他操作觸發重新執行 (rerun)
        at.run()

        # 驗證：API 呼叫次數依然為 1，未產生任何重複呼叫
        self.assertEqual(mock_generate.call_count, 1, "Streamlit 重新渲染時不應重複呼叫 API")
        self.assertIn("分析結果：優質高蛋白。", " ".join([m.value for m in at.markdown]))

    # ------------------------------------------------------------------
    # 測試 7：輸入資料驗證失敗。
    # 預期：不執行 Random Forest 預測，也不呼叫營養觀察 API。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    @patch('utils.model_runner.predict_nutrition_grade')
    def test_07_validation_failure_blocks_rf_and_llm(self, mock_rf, mock_llm):
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        # 輸入負數使驗證失敗
        at.number_input[0].set_value(200.0)
        at.number_input[1].set_value(-5.0) # 負數
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(30.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.5)
        at.button[0].click().run()

        # 驗證兩者皆未被呼叫
        mock_rf.assert_not_called()
        mock_llm.assert_not_called()

    # ------------------------------------------------------------------
    # 測試 8：營養觀察服務持續不可用。
    # 預期：重試次數有限，不會無限等待；使用者仍能看到清楚的狀態提示。
    # ------------------------------------------------------------------
    @patch('google.genai.models.Models.generate_content')
    def test_08_persistent_failure_bounded_retries(self, mock_generate):
        mock_generate.side_effect = TimeoutError("Connection timed out after 30s")

        advice, error_info = generate_nutrition_advice(
            self.dummy_nutrition, grade='C', max_retries=3, initial_delay=0.01
        )
        self.assertIsNone(advice)
        self.assertIsNotNone(error_info)
        # 應剛好重試 max_retries (3) 次，不會無窮迴圈
        self.assertEqual(mock_generate.call_count, 3)
        self.assertIn(error_info['category'], ['transient_service', 'unknown'])

if __name__ == '__main__':
    unittest.main()

import os
import sys
import unittest
from streamlit.testing.v1 import AppTest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(CURRENT_DIR)
APP_FILE = os.path.join(DEMO_DIR, 'app.py')

class TestStreamlitAppUI(unittest.TestCase):

    def test_ui_case_1_all_zeros_warning(self):
        """測試 UI 案例 1：填入全部為 0，未勾選確認時，應顯示警告，不應出現分級或營養觀察"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        
        # 填入六項 0
        at.number_input[0].set_value(0.0) # 熱量
        at.number_input[1].set_value(0.0) # 蛋白質
        at.number_input[2].set_value(0.0) # 脂肪
        at.number_input[3].set_value(0.0) # 碳水
        at.number_input[4].set_value(0.0) # 糖
        at.number_input[5].set_value(0.0) # 食鹽
        
        # 點擊確認送出
        at.button[0].click().run()

        # 檢查警告訊息
        warnings_text = [w.value for w in at.warning]
        self.assertTrue(any("目前六項營養數值皆為 0" in w for w in warnings_text), f"應顯示全零警告，但目前警告為: {warnings_text}")
        
        # 確認沒有顯示 Nutrition Grade 標題或分級
        markdown_text = [m.value for m in at.markdown]
        self.assertFalse(any("Nutrition Grade" in m for m in markdown_text), "全零未確認狀態下不得顯示 Nutrition Grade")

    def test_ui_case_2_missing_field(self):
        """測試 UI 案例 2：任一欄位留空缺失，阻止預測並指出缺失欄位"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        
        # 僅填寫部分欄位，蛋白質與脂肪留空 None
        at.number_input[0].set_value(200.0)
        # number_input[1] 蛋白質 保持為 None
        at.number_input[2].set_value(None)
        at.number_input[3].set_value(30.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.5)

        at.button[0].click().run()

        errors_text = [e.value for e in at.error]
        markdown_text = [m.value for m in at.markdown]
        all_err_content = " ".join(errors_text + markdown_text)
        self.assertTrue("蛋白質" in all_err_content or "缺少必要欄位" in all_err_content, "應指出缺失欄位")
        self.assertFalse(any("Nutrition Grade" in m for m in markdown_text))

    def test_ui_case_3_negative_value(self):
        """測試 UI 案例 3：其中一項數值為負數，顯示錯誤訊息並阻止預測"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()
        
        at.number_input[0].set_value(200.0)
        at.number_input[1].set_value(5.0)
        at.number_input[2].set_value(-3.0) # 脂肪為負數
        at.number_input[3].set_value(20.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.2)

        at.button[0].click().run()

        errors_text = [e.value for e in at.error]
        markdown_text = [m.value for m in at.markdown]
        all_err_content = " ".join(errors_text + markdown_text)
        self.assertTrue("不可為負數" in all_err_content and "脂肪" in all_err_content)
        self.assertFalse(any("Nutrition Grade" in m for m in markdown_text))

    def test_ui_case_5_sugar_greater_than_carbs(self):
        """測試 UI 案例 5：糖高於碳水化合物，顯示提示並阻止預測"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()

        at.number_input[0].set_value(250.0)
        at.number_input[1].set_value(5.0)
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(10.0)  # 碳水化合物 10g
        at.number_input[4].set_value(20.0)  # 糖 20g (高於碳水)
        at.number_input[5].set_value(0.2)

        at.button[0].click().run()

        markdown_text = [m.value for m in at.markdown]
        all_err_content = " ".join([e.value for e in at.error] + markdown_text)
        self.assertTrue("糖與碳水化合物" in all_err_content and "高於" in all_err_content)
        self.assertFalse(any("Nutrition Grade" in m for m in markdown_text))

    def test_ui_case_6_extreme_value(self):
        """測試 UI 案例 6：極端數值（熱量 > 1000 kcal），提示確認並阻止預測"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()

        at.number_input[0].set_value(2000.0) # 極端熱量
        at.number_input[1].set_value(5.0)
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(20.0)
        at.number_input[4].set_value(5.0)
        at.number_input[5].set_value(0.2)

        at.button[0].click().run()

        all_err_content = " ".join([e.value for e in at.error] + [m.value for m in at.markdown])
        self.assertTrue("熱量極端異常" in all_err_content)
        self.assertFalse(any("Nutrition Grade" in m for m in [m.value for m in at.markdown]))

    def test_ui_case_7_valid_inputs(self):
        """測試 UI 案例 7：合理數值且基準正確，正常執行預測並顯示分級"""
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()

        at.number_input[0].set_value(350.0)
        at.number_input[1].set_value(8.0)
        at.number_input[2].set_value(12.0)
        at.number_input[3].set_value(55.0)
        at.number_input[4].set_value(10.0)
        at.number_input[5].set_value(0.8)

        at.button[0].click().run()

        markdown_text = [m.value for m in at.markdown]
        self.assertTrue(any("Nutrition Grade" in m for m in markdown_text), "合理數值通過後應顯示 Nutrition Grade")

    def test_ui_case_8_and_9_session_state_transitions(self):
        """測試 UI 案例 8 & 9：
        - 案例 8：先異常再修正成正常 -> 重新預測成功。
        - 案例 9：先正常再改成異常 -> 必須清除舊預測結果，不可殘留顯示舊結果。
        """
        at = AppTest.from_file(APP_FILE, default_timeout=30).run()

        # Step A: 輸入正常資料
        at.number_input[0].set_value(300.0)
        at.number_input[1].set_value(10.0)
        at.number_input[2].set_value(5.0)
        at.number_input[3].set_value(50.0)
        at.number_input[4].set_value(10.0)
        at.number_input[5].set_value(0.5)
        at.button[0].click().run()

        # 確認正常顯示結果
        self.assertTrue(any("Nutrition Grade" in m for m in [m.value for m in at.markdown]))

        # Step B (測試案例 9): 將脂肪改為負數 -8.0，再次送出
        at.number_input[2].set_value(-8.0)
        at.button[0].click().run()

        # 預期：舊結果已被清除，顯示錯誤訊息，絕無 Nutrition Grade
        markdown_text_after_invalid = [m.value for m in at.markdown]
        self.assertFalse(any("Nutrition Grade" in m for m in markdown_text_after_invalid), "改成異常資料後舊結果必須被清除，不得殘留")
        all_err = " ".join([e.value for e in at.error] + markdown_text_after_invalid)
        self.assertTrue("不可為負數" in all_err and "脂肪" in all_err)

        # Step C (測試案例 8): 修正回正常數值 8.0，再次送出
        at.number_input[2].set_value(8.0)
        at.button[0].click().run()

        # 預期：重新產生最新預測結果
        markdown_text_after_fixed = [m.value for m in at.markdown]
        self.assertTrue(any("Nutrition Grade" in m for m in markdown_text_after_fixed), "修正回正常後應重新顯示最新預測")

if __name__ == '__main__':
    unittest.main()

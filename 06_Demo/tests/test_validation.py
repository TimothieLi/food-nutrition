import os
import sys
import unittest
import math

# 加入 06_Demo 至路徑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(DEMO_DIR)

from utils.validator import validate_nutrition_data
from utils.model_runner import load_model, predict_nutrition_grade

class TestNutritionValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 載入實際 Random Forest 模型確認可正常連動
        cls.model = load_model()

    def test_case_1_all_zeros(self):
        """測試案例 1：六項營養數值全部為 0。
        預期：顯示需要確認的警告，不直接輸出 A～E 級，也不產生營養觀察。
        """
        all_zero_data = {
            'energy_kcal': 0.0,
            'proteins_100g': 0.0,
            'fat_100g': 0.0,
            'carbohydrates_100g': 0.0,
            'sugars_100g': 0.0,
            'salt_100g': 0.0
        }
        # 未勾選確認時
        result = validate_nutrition_data(all_zero_data, basis='per_100g', confirmed_all_zero=False)
        self.assertEqual(result['status'], 'warning')
        self.assertFalse(result['is_valid'], "未經確認之全零數值不可直接進入預測")
        self.assertIsNone(result['cleaned_data'], "未通過驗證時 cleaned_data 應為 None")
        self.assertTrue(any("皆為 0" in w for w in result['warnings']), "應包含六項數值皆為 0 之警告")

        # 若使用者勾選確認（如確認為純水或無熱量茶）
        result_confirmed = validate_nutrition_data(all_zero_data, basis='per_100g', confirmed_all_zero=True)
        self.assertEqual(result_confirmed['status'], 'valid')
        self.assertTrue(result_confirmed['is_valid'])
        self.assertTrue(result_confirmed['info']['all_zero'])

    def test_case_2_missing_field(self):
        """測試案例 2：其中一項營養數值缺失（None 或未填）。
        預期：指出缺失欄位，阻止預測。
        """
        data_missing = {
            'energy_kcal': 250.0,
            'proteins_100g': None,  # 缺失
            'fat_100g': 10.0,
            'carbohydrates_100g': 30.0,
            'sugars_100g': 5.0,
            'salt_100g': 0.5
        }
        result = validate_nutrition_data(data_missing, basis='per_100g')
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['is_valid'])
        self.assertIsNone(result['cleaned_data'])
        self.assertTrue(any("蛋白質" in err for err in result['errors']), "錯誤訊息應明確指明缺失的【蛋白質】欄位")

    def test_case_3_negative_number(self):
        """測試案例 3：其中一項營養數值為負數。
        預期：顯示錯誤訊息，阻止預測。
        """
        data_negative = {
            'energy_kcal': 200.0,
            'proteins_100g': 5.0,
            'fat_100g': -2.5,  # 負數
            'carbohydrates_100g': 20.0,
            'sugars_100g': 5.0,
            'salt_100g': 0.2
        }
        result = validate_nutrition_data(data_negative, basis='per_100g')
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['is_valid'])
        self.assertIsNone(result['cleaned_data'])
        self.assertTrue(any("不可為負數" in err and "脂肪" in err for err in result['errors']), "應明確指出脂肪數值不可為負數")

    def test_case_4_invalid_format_or_nan(self):
        """測試案例 4：其中一項數值為無法解析的文字或 NaN。
        預期：妥善處理錯誤，不讓程式崩潰，也不呼叫模型。
        """
        # 測試文字
        data_text = {
            'energy_kcal': "abc_error",
            'proteins_100g': 5.0,
            'fat_100g': 3.0,
            'carbohydrates_100g': 15.0,
            'sugars_100g': 2.0,
            'salt_100g': 0.1
        }
        result_text = validate_nutrition_data(data_text, basis='per_100g')
        self.assertEqual(result_text['status'], 'invalid')
        self.assertFalse(result_text['is_valid'])
        self.assertTrue(any("無法轉換為有效數字" in err and "熱量" in err for err in result_text['errors']))

        # 測試 NaN
        data_nan = {
            'energy_kcal': 100.0,
            'proteins_100g': float('nan'),
            'fat_100g': 3.0,
            'carbohydrates_100g': 15.0,
            'sugars_100g': 2.0,
            'salt_100g': 0.1
        }
        result_nan = validate_nutrition_data(data_nan, basis='per_100g')
        self.assertEqual(result_nan['status'], 'invalid')
        self.assertFalse(result_nan['is_valid'])
        self.assertTrue(any("NaN" in err and "蛋白質" in err for err in result_nan['errors']))

    def test_case_5_sugar_greater_than_carbohydrates(self):
        """測試案例 5：糖含量高於碳水化合物含量。
        預期：顯示合理的檢查提示，依照設定的驗證規則決定是否允許預測（阻擋預測）。
        """
        data_sugar_high = {
            'energy_kcal': 300.0,
            'proteins_100g': 2.0,
            'fat_100g': 5.0,
            'carbohydrates_100g': 10.0,  # 碳水化合物 10g
            'sugars_100g': 25.0,          # 糖 25g (不合理)
            'salt_100g': 0.1
        }
        result = validate_nutrition_data(data_sugar_high, basis='per_100g')
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['is_valid'])
        self.assertTrue(any("糖含量" in err and "高於碳水化合物" in err for err in result['errors']))

    def test_case_6_extreme_values(self):
        """測試案例 6：輸入數值極大，可能存在單位或 OCR 辨識錯誤。
        預期：提示使用者確認，不應無條件接受或任意修改數值。
        """
        # 熱量超過理論極限 (> 1000 kcal / 100g)
        data_extreme_energy = {
            'energy_kcal': 2500.0,  # 極端值 (純油亦僅約 900 kcal)
            'proteins_100g': 10.0,
            'fat_100g': 20.0,
            'carbohydrates_100g': 30.0,
            'sugars_100g': 5.0,
            'salt_100g': 0.5
        }
        result_energy = validate_nutrition_data(data_extreme_energy, basis='per_100g')
        self.assertEqual(result_energy['status'], 'invalid')
        self.assertFalse(result_energy['is_valid'])
        self.assertTrue(any("熱量極端異常" in err for err in result_energy['errors']))

        # 單項營養素超過 100g (例如脂肪 150g)
        data_extreme_fat = {
            'energy_kcal': 500.0,
            'proteins_100g': 5.0,
            'fat_100g': 150.0,  # 每 100g 食品脂肪不可能達 150g
            'carbohydrates_100g': 10.0,
            'sugars_100g': 5.0,
            'salt_100g': 0.5
        }
        result_fat = validate_nutrition_data(data_extreme_fat, basis='per_100g')
        self.assertEqual(result_fat['status'], 'invalid')
        self.assertFalse(result_fat['is_valid'])
        self.assertTrue(any("超過每 100g 食品物理上限" in err for err in result_fat['errors']))

    def test_case_7_valid_reasonable_values(self):
        """測試案例 7：輸入六項合理且大於或等於零的營養數值，並確認數值基準正確。
        預期：可以正常執行原本的模型預測流程，顯示 A～E 分級。
        """
        data_valid = {
            'energy_kcal': 380.0,
            'proteins_100g': 7.5,
            'fat_100g': 12.0,
            'carbohydrates_100g': 60.0,
            'sugars_100g': 15.0,
            'salt_100g': 0.8
        }
        result = validate_nutrition_data(data_valid, basis='per_100g')
        self.assertEqual(result['status'], 'valid')
        self.assertTrue(result['is_valid'])
        self.assertIsNotNone(result['cleaned_data'])

        # 實際送入模型預測
        grade = predict_nutrition_grade(self.model, result['cleaned_data'])
        self.assertIn(grade, ['A', 'B', 'C', 'D', 'E'], f"預測結果應為 A~E 分級，得到 {grade}")

    def test_case_8_workflow_invalid_then_corrected(self):
        """測試案例 8：第一次輸入異常資料，第二次修正成合理資料。
        預期：第一次不產生有效預測；第二次通過驗證後，應根據最新數值重新預測，不可沿用先前的錯誤結果。
        """
        # 模擬 Streamlit session_state
        state = {'prediction_result': None}

        # 第一次：輸入異常（缺少糖含量）
        bad_input = {
            'energy_kcal': 350.0,
            'proteins_100g': 6.0,
            'fat_100g': 10.0,
            'carbohydrates_100g': 50.0,
            'sugars_100g': None,  # 異常缺失
            'salt_100g': 0.5
        }
        res1 = validate_nutrition_data(bad_input, basis='per_100g')
        state['prediction_result'] = None  # 每次送出均清空
        if res1['is_valid']:
            state['prediction_result'] = predict_nutrition_grade(self.model, res1['cleaned_data'])
        
        self.assertIsNone(state['prediction_result'], "第一次異常資料不可產生預測結果")

        # 第二次：修正糖含量為 8.0g
        corrected_input = bad_input.copy()
        corrected_input['sugars_100g'] = 8.0

        res2 = validate_nutrition_data(corrected_input, basis='per_100g')
        state['prediction_result'] = None
        if res2['is_valid']:
            state['prediction_result'] = predict_nutrition_grade(self.model, res2['cleaned_data'])

        self.assertIsNotNone(state['prediction_result'], "第二次修正後應成功產生預測結果")
        self.assertIn(state['prediction_result'], ['A', 'B', 'C', 'D', 'E'])

    def test_case_9_workflow_valid_then_abnormal(self):
        """測試案例 9：第一次輸入合理資料，第二次改成異常資料。
        預期：第二次必須停止預測並清除或隱藏舊的分析結果，不可繼續顯示第一次的結果，讓使用者誤以為那是最新預測。
        """
        # 模擬 Streamlit session_state
        state = {'prediction_result': None}

        # 第一次：輸入合理資料
        valid_input = {
            'energy_kcal': 220.0,
            'proteins_100g': 4.0,
            'fat_100g': 2.0,
            'carbohydrates_100g': 45.0,
            'sugars_100g': 12.0,
            'salt_100g': 0.3
        }
        res1 = validate_nutrition_data(valid_input, basis='per_100g')
        state['prediction_result'] = None
        if res1['is_valid']:
            state['prediction_result'] = predict_nutrition_grade(self.model, res1['cleaned_data'])

        first_grade = state['prediction_result']
        self.assertIsNotNone(first_grade, "第一次應成功產生預測")

        # 第二次：改成異常資料（例如脂肪輸入為負數 -10.0）
        abnormal_input = valid_input.copy()
        abnormal_input['fat_100g'] = -10.0

        # 送出表單：立即重置 prediction_result
        state['prediction_result'] = None
        res2 = validate_nutrition_data(abnormal_input, basis='per_100g')
        if res2['is_valid']:
            state['prediction_result'] = predict_nutrition_grade(self.model, res2['cleaned_data'])

        # 驗證第二次結果
        self.assertFalse(res2['is_valid'])
        self.assertIsNone(state['prediction_result'], "第二次改為異常資料後，必須清除舊結果，不可保留或顯示第一次的預測")

if __name__ == '__main__':
    unittest.main()

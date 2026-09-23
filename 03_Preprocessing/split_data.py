import pandas as pd
from sklearn.model_selection import train_test_split

def split_data():
    data_path = '../01_Data/processed/clean_food_data.csv'
    
    print("Loading clean data...")
    df = pd.read_csv(data_path)
    total_len = len(df)
    
    label_col = 'nutrition_grade_fr'
    
    # 第一次切分：Train (70%) vs Temp (30%)
    df_train, df_temp = train_test_split(
        df, 
        test_size=0.30, 
        random_state=42, 
        stratify=df[label_col]
    )
    
    # 第二次切分：Temp (30%) 拆成 Validation (15%) 與 Test (15%)
    # 佔 Temp 的 50%
    df_val, df_test = train_test_split(
        df_temp, 
        test_size=0.50, 
        random_state=42, 
        stratify=df_temp[label_col]
    )
    
    print("Saving datasets...")
    df_train.to_csv('../01_Data/processed/train.csv', index=False)
    df_val.to_csv('../01_Data/processed/validation.csv', index=False)
    df_test.to_csv('../01_Data/processed/test.csv', index=False)
    
    # 統計結果
    def get_stats(data):
        counts = data[label_col].value_counts().to_dict()
        pcts = data[label_col].value_counts(normalize=True).to_dict()
        return counts, pcts
        
    orig_counts, orig_pcts = get_stats(df)
    train_counts, train_pcts = get_stats(df_train)
    val_counts, val_pcts = get_stats(df_val)
    test_counts, test_pcts = get_stats(df_test)
    
    report = f"""# Train / Validation / Test Split Report

## 1. 資料筆數統計
- **原始資料筆數**: {total_len:,} 筆
- **Train (70%)**: {len(df_train):,} 筆
- **Validation (15%)**: {len(df_val):,} 筆
- **Test (15%)**: {len(df_test):,} 筆

## 2. 類別分布 (A/B/C/D/E)

### 數量分布
| 類別 | Original 數量 | Train 數量 | Validation 數量 | Test 數量 |
| :--- | :--- | :--- | :--- | :--- |
| **A** | {orig_counts.get('a', 0):,} | {train_counts.get('a', 0):,} | {val_counts.get('a', 0):,} | {test_counts.get('a', 0):,} |
| **B** | {orig_counts.get('b', 0):,} | {train_counts.get('b', 0):,} | {val_counts.get('b', 0):,} | {test_counts.get('b', 0):,} |
| **C** | {orig_counts.get('c', 0):,} | {train_counts.get('c', 0):,} | {val_counts.get('c', 0):,} | {test_counts.get('c', 0):,} |
| **D** | {orig_counts.get('d', 0):,} | {train_counts.get('d', 0):,} | {val_counts.get('d', 0):,} | {test_counts.get('d', 0):,} |
| **E** | {orig_counts.get('e', 0):,} | {train_counts.get('e', 0):,} | {val_counts.get('e', 0):,} | {test_counts.get('e', 0):,} |

### 百分比分布 (Stratified Check)
| 類別 | Original 比例 | Train 比例 | Validation 比例 | Test 比例 |
| :--- | :--- | :--- | :--- | :--- |
| **A** | {orig_pcts.get('a', 0):.2%} | {train_pcts.get('a', 0):.2%} | {val_pcts.get('a', 0):.2%} | {test_pcts.get('a', 0):.2%} |
| **B** | {orig_pcts.get('b', 0):.2%} | {train_pcts.get('b', 0):.2%} | {val_pcts.get('b', 0):.2%} | {test_pcts.get('b', 0):.2%} |
| **C** | {orig_pcts.get('c', 0):.2%} | {train_pcts.get('c', 0):.2%} | {val_pcts.get('c', 0):.2%} | {test_pcts.get('c', 0):.2%} |
| **D** | {orig_pcts.get('d', 0):.2%} | {train_pcts.get('d', 0):.2%} | {val_pcts.get('d', 0):.2%} | {test_pcts.get('d', 0):.2%} |
| **E** | {orig_pcts.get('e', 0):.2%} | {train_pcts.get('e', 0):.2%} | {val_pcts.get('e', 0):.2%} | {test_pcts.get('e', 0):.2%} |

## 3. 確認事項
- [x] **Stratified Split**: 上述表格顯示，Train, Validation, Test 三個資料集的類別比例與原始資料集幾乎完全一致 (差距在小數點後位數以內)，成功保持分布平衡。
- [x] **資料獨立性**: 已使用 `train_test_split` 強制切分，切分後的 Train/Val/Test 互相獨立不重疊。
- [x] **Random Seed**: 已設定 `random_state=42` 確保每次切分結果一致。
"""

    with open('split_report.md', 'w') as f:
        f.write(report)
        
    print("Split complete. Report generated.")

if __name__ == "__main__":
    split_data()

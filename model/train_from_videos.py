# -*- coding: utf-8 -*-
"""
真實影片模型訓練腳本 (Real Video Training Script)
功能：讀取 training_data 資料夾中的影片，提取特徵，訓練出真實的 AI 模型。
"""

import os
import glob
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

# 添加專案根目錄到 path 以便匯入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.processor import process_video

# 設定路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'training_data')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'rehab_model.pkl')
TEMP_CSV_DIR = os.path.join(BASE_DIR, 'training_data', 'temp_features')

if not os.path.exists(TEMP_CSV_DIR):
    os.makedirs(TEMP_CSV_DIR)

def train_from_videos():
    print("=========================================")
    print("   開始執行真實影片訓練流程")
    print("=========================================")
    
    all_data = []
    
    # 定義類別與對應資料夾
    categories = {
        0: '0_healthy',
        1: '1_frailty',
        2: '2_compensation'
    }
    
    for label, folder_name in categories.items():
        folder_path = os.path.join(TRAIN_DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"[警告] 資料夾不存在: {folder_path}，跳過。")
            continue
            
        # 搜尋所有格式影片
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        print(f"\n正在處理類別 {label} ({folder_name}): 找到 {len(video_files)} 個影片")
        
        for vid_path in video_files:
            filename = os.path.basename(vid_path)
            print(f"  -> 分析中: {filename} ...", end=" ")
            
            # 使用我們現有的 process_video 提取特徵
            # 我們不需要保留詳細的 CSV，所以存到暫存區或覆蓋都行
            temp_csv_path = os.path.join(TEMP_CSV_DIR, f"{filename}.csv")
            try:
                df, features = process_video(vid_path, temp_csv_path)
                
                if features:
                    # 加入 Label
                    features['Label'] = label
                    all_data.append(features)
                    print("[成功]")
                else:
                    print("[失敗 - 無法提取骨架]")
            except Exception as e:
                print(f"[錯誤: {e}]")

    # 檢查是否有資料
    if not all_data:
        print("\n[錯誤] 沒有成功提取到任何資料，請檢查 training_data 資料夾內是否有影片。")
        return

    # 建立 DataFrame
    train_df = pd.DataFrame(all_data)
    
    # 準備訓練特徵 (必須與 app.py 預測時的輸入一致)
    feature_cols = ['X1_MaxTrunkLean', 'X2_TotalDuration', 'X3_MinHandKneeDist', 'X4_EndKneeAngle']
    X = train_df[feature_cols]
    y = train_df['Label']
    
    print("\n-----------------------------------------")
    print(f"資料集準備完成，共 {len(train_df)} 筆樣本")
    print(train_df.groupby('Label').size())
    print("-----------------------------------------")

    # 訓練模型
    print("開始訓練決策樹模型 (Decision Tree)...")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    # 簡單驗證 (Training Accuracy)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"訓練準確率 (Training Accuracy): {acc*100:.2f}%")
    
    # 儲存模型
    joblib.dump(clf, MODEL_PATH)
    print(f"\n[成功] 新模型已儲存至: {MODEL_PATH}")
    print("現在 app.py 將會使用從這些影片學到的新邏輯進行判斷！")

if __name__ == "__main__":
    train_from_videos()

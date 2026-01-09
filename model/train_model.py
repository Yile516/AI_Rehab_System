# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:23:41 2025

@author: ivanl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier

# 設定存檔路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'rehab_model.pkl')

# --- 定義 STS 測試的專家規則 (臨床標準) ---
# Class 0: 健康 (Healthy) - 站得快、手抱胸(距離遠)、前傾適中
# Class 1: 虛弱 (Frail) - 站得慢、前傾很大(為了甩上去)、手抱胸
# Class 2: 代償 (Compensated) - 手扶膝蓋(距離近)

def train():
    print("--- 正在生成復健模擬數據 (Synthetic Data) ---")
    data = []
    
    # 生成 100 筆健康數據
    for _ in range(100):
        data.append({
            'X1_MaxTrunkLean': np.random.normal(30, 5),   # 正常前傾約30度
            'X2_TotalDuration': np.random.normal(1.5, 0.3), # 動作快
            'X3_MinHandKneeDist': np.random.normal(0.4, 0.1), # 手離膝蓋遠
            'X4_EndKneeAngle': np.random.normal(175, 5),    # 站得直
            'Label': 0
        })

    # 生成 100 筆虛弱數據 (高風險)
    for _ in range(100):
        data.append({
            'X1_MaxTrunkLean': np.random.normal(60, 8),   # 過度前傾 (>45)
            'X2_TotalDuration': np.random.normal(3.5, 0.8), # 動作慢
            'X3_MinHandKneeDist': np.random.normal(0.35, 0.1), # 手還是在身上
            'X4_EndKneeAngle': np.random.normal(165, 8),    # 可能站不直
            'Label': 1
        })

    # 生成 100 筆代償數據 (扶膝蓋)
    for _ in range(100):
        data.append({
            'X1_MaxTrunkLean': np.random.normal(45, 10),
            'X2_TotalDuration': np.random.normal(2.5, 0.5),
            'X3_MinHandKneeDist': np.random.normal(0.05, 0.02), # 關鍵！手非常靠近膝蓋
            'X4_EndKneeAngle': np.random.normal(170, 5),
            'Label': 2
        })

    df = pd.DataFrame(data)
    X = df.drop('Label', axis=1)
    y = df['Label']

    print("--- 訓練決策樹模型 ---")
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"模型已儲存至: {MODEL_PATH}")

if __name__ == "__main__":
    train()
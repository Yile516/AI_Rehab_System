# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:23:41 2025
Updated: 2026-06-10 - 依據真實標記影片的特徵值重新錨定模擬數據分佈

@author: ivanl

真實量測錨點 (Real-video Anchors)
--------------------------------------------------
影片              類別      X1_Trunk  X2_Dur  X3_HKD   X4_Knee
video2.mp4       健康 (0)   52.10°   3.80s   0.302    156.94°
healthy.mp4      健康 (0)   13.57°   2.76s   0.145    165.00°
frailty.mp4      衰弱 (1)   68.56°   6.07s   0.290    159.76°
compensation.mp4 代償 (2)   66.60°   4.58s   0.119    159.07°
--------------------------------------------------

分類關鍵決策依據：
  - X2_TotalDuration 是最強鑑別特徵：
      健康 < 4s  |  代償 4~5.5s  |  衰弱 > 5.5s
  - X3_AttHandKneeDist 次要鑑別：
      代償 < 0.18 (手扶膝蓋) vs 健康/衰弱 > 0.22
  - X1_MaxTrunkLean 輔助特徵：
      健康抱胸型 ~52°  vs  一般健康型 ~14°  vs  衰弱/代償 ~65~70°
"""

import io
import sys
import pandas as pd
import numpy as np
import joblib
import os

# 修正 Windows 終端機編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

# 設定存檔路徑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'rehab_model.pkl')

np.random.seed(42)

# ──────────────────────────────────────────────────────────
# 真實影片量測錨點 (會直接加入訓練集，並設定較高的複製倍數)
# ──────────────────────────────────────────────────────────
REAL_ANCHORS = [
    # 健康 - 抱胸型 (video2.mp4)
    {'X1_MaxTrunkLean': 52.10, 'X2_TotalDuration': 3.80,
     'X3_AttHandKneeDist': 0.302, 'X4_EndKneeAngle': 156.94, 'Label': 0},
    # 健康 - 一般型 (healthy.mp4 / 健康.mp4)
    {'X1_MaxTrunkLean': 43.45, 'X2_TotalDuration': 2.76,
     'X3_AttHandKneeDist': 0.2264, 'X4_EndKneeAngle': 157.90, 'Label': 0},
    # 衰弱 (frailty.mp4)
    {'X1_MaxTrunkLean': 68.56, 'X2_TotalDuration': 6.07,
     'X3_AttHandKneeDist': 0.290, 'X4_EndKneeAngle': 159.76, 'Label': 1},
    # 代償 (compensation.mp4 / 代償.mp4)
    {'X1_MaxTrunkLean': 66.60, 'X2_TotalDuration': 4.58,
     'X3_AttHandKneeDist': 0.119, 'X4_EndKneeAngle': 159.07, 'Label': 2},
]


def generate_synthetic_data(n_per_class=200):
    """
    以真實量測值為中心，生成模擬擴充訓練數據。
    每個類別的分佈皆錨定於真實影片觀測值。
    """
    data = []

    # ── Class 0: 健康 ──────────────────────────────────────
    # 健康樣本分為兩個子群（抱胸型 & 一般標準型），各佔 50%
    for _ in range(n_per_class // 2):
        # 子群 A：一般標準健康 (典型快速起立，前傾少)
        data.append({
            'X1_MaxTrunkLean':    max(5, np.random.normal(18, 7)),     # 錨點 13.57°，上限可到 30
            'X2_TotalDuration':   max(1.0, np.random.normal(2.8, 0.5)), # 錨點 2.76s
            'X3_AttHandKneeDist': np.clip(np.random.normal(0.20, 0.06), 0.10, 0.40),
            'X4_EndKneeAngle':    np.clip(np.random.normal(165, 6), 145, 180),
            'Label': 0
        })

    for _ in range(n_per_class // 2):
        # 子群 B：抱胸健康型 (前傾較大但動作流暢，手遠離膝蓋)
        data.append({
            'X1_MaxTrunkLean':    np.clip(np.random.normal(50, 6), 38, 65),  # 錨點 52.10°
            'X2_TotalDuration':   np.clip(np.random.normal(3.8, 0.5), 2.5, 4.9), # 錨點 3.80s
            'X3_AttHandKneeDist': np.clip(np.random.normal(0.30, 0.05), 0.20, 0.45),  # 錨點 0.302，手遠離膝蓋
            'X4_EndKneeAngle':    np.clip(np.random.normal(158, 6), 140, 178),
            'Label': 0
        })

    # ── Class 1: 衰弱風險 ──────────────────────────────────
    # 特徵：動作非常慢(>5.5s)、大幅前傾、手離膝有一定距離
    for _ in range(n_per_class):
        data.append({
            'X1_MaxTrunkLean':    np.clip(np.random.normal(68, 6), 50, 85),    # 錨點 68.56°
            'X2_TotalDuration':   np.clip(np.random.normal(6.1, 0.8), 4.8, 9.0), # 錨點 6.07s，明顯長
            'X3_AttHandKneeDist': np.clip(np.random.normal(0.28, 0.05), 0.18, 0.42), # 錨點 0.290，手不在膝蓋上
            'X4_EndKneeAngle':    np.clip(np.random.normal(160, 7), 140, 178),
            'Label': 1
        })

    # ── Class 2: 代償動作 (扶膝蓋) ─────────────────────────
    # 特徵：手非常靠近膝蓋(X3低)、動作中等偏慢、大幅前傾
    for _ in range(n_per_class):
        data.append({
            'X1_MaxTrunkLean':    np.clip(np.random.normal(65, 8), 45, 82),    # 錨點 66.60°
            'X2_TotalDuration':   np.clip(np.random.normal(4.6, 0.7), 3.0, 6.5), # 錨點 4.58s
            'X3_AttHandKneeDist': np.clip(np.random.normal(0.12, 0.03), 0.04, 0.19), # 錨點 0.119！手非常靠近膝蓋
            'X4_EndKneeAngle':    np.clip(np.random.normal(159, 7), 140, 178),
            'Label': 2
        })

    return data


def add_real_anchors_with_jitter(anchors, n_jitter=30):
    """
    將真實錨點直接加入訓練集，並在其周圍加入少量抖動
    以增加真實樣本的影響力，避免被大量合成數據稀釋。
    """
    aug_data = []
    jitter_scales = {
        'X1_MaxTrunkLean':    1.5,
        'X2_TotalDuration':   0.15,
        'X3_AttHandKneeDist': 0.01,
        'X4_EndKneeAngle':    2.0,
    }
    for anchor in anchors:
        # 原始真實點加入一次
        aug_data.append(dict(anchor))
        # 加入 n_jitter 筆小抖動樣本
        for _ in range(n_jitter):
            jittered = {}
            for feat, scale in jitter_scales.items():
                jittered[feat] = anchor[feat] + np.random.normal(0, scale)
            jittered['Label'] = anchor['Label']
            aug_data.append(jittered)
    return aug_data


def train():
    print("=" * 55)
    print("   開始訓練 AI 模型 (真實數據錨點版本)")
    print("=" * 55)

    # 1. 生成錨定真實量測值的合成數據
    print("\n[1/4] 生成以真實影片特徵為錨點的擴充合成數據...")
    synthetic_data = generate_synthetic_data(n_per_class=200)

    # 2. 加入真實影片量測值（加強抖動擴充）
    print("[2/4] 將真實影片量測錨點加入訓練集 (含小幅抖動擴充)...")
    real_data = add_real_anchors_with_jitter(REAL_ANCHORS, n_jitter=30)

    # 3. 合併並建立 DataFrame
    all_data = synthetic_data + real_data
    df = pd.DataFrame(all_data)

    feature_cols = ['X1_MaxTrunkLean', 'X2_TotalDuration', 'X3_AttHandKneeDist', 'X4_EndKneeAngle']
    X = df[feature_cols]
    y = df['Label']

    print(f"   合計樣本數: {len(df)}")
    print(f"   類別分布: {dict(y.value_counts().sort_index())}")
    print(f"   (0=健康, 1=衰弱, 2=代償)")

    # 4. 訓練/測試分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 訓練 Decision Tree 模型以保持高可解釋性
    print("\n[3/4] 訓練 Decision Tree 模型...")
    clf = DecisionTreeClassifier(
        max_depth=4,
        random_state=42,
        class_weight='balanced'
    )

    # 分層 5 折交叉驗證
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf)
    print(f"   5 折交叉驗證準確率: {cv_scores.mean()*100:.2f}% (Std: ±{cv_scores.std()*100:.2f}%)")

    # 最終訓練
    clf.fit(X_train, y_train)

    # 6. 測試集評估
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average='weighted')

    print(f"\n[4/4] 測試集評估結果:")
    print(f"   準確率 (Accuracy): {test_acc*100:.2f}%")
    print(f"   F1-score (Weighted): {test_f1*100:.2f}%")
    print(f"\n   混淆矩陣 (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {'':10s} 預測:健康  預測:衰弱  預測:代償")
    class_names = ['健康', '衰弱', '代償']
    for i, row in enumerate(cm):
        print(f"   實際:{class_names[i]:4s}  {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")

    print(f"\n   分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['健康', '衰弱', '代償']))

    # 7. 驗證真實錨點是否正確預測
    print("\n   === 驗證真實影片錨點預測結果 ===")
    anchor_df = pd.DataFrame([
        {**{k: v for k, v in a.items() if k != 'Label'},
         'Video': ['video2.mp4', 'healthy.mp4', 'frailty.mp4', 'compensation.mp4'][i],
         'TrueLabel': a['Label']}
        for i, a in enumerate(REAL_ANCHORS)
    ])
    X_anchors = anchor_df[feature_cols]
    y_anchor_pred = clf.predict(X_anchors)
    y_anchor_prob = clf.predict_proba(X_anchors)
    label_map = {0: '健康', 1: '衰弱', 2: '代償'}
    for i, row in anchor_df.iterrows():
        true_lbl = label_map[int(row['TrueLabel'])]
        pred_lbl = label_map[y_anchor_pred[i]]
        prob_str = f"[健康:{y_anchor_prob[i][0]:.2f} 衰弱:{y_anchor_prob[i][1]:.2f} 代償:{y_anchor_prob[i][2]:.2f}]"
        status = "[正確]" if y_anchor_pred[i] == row['TrueLabel'] else "[錯誤]"
        print(f"   {status} {row['Video']:20s} 真實:{true_lbl} -> 預測:{pred_lbl} {prob_str}")

    # 8. 特徵重要性
    print("\n   === 特徵重要性 (Feature Importance) ===")
    importances = clf.feature_importances_
    for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        print(f"   {feat:25s}: {imp:.4f}  {bar}")

    # 9. 儲存模型
    joblib.dump(clf, MODEL_PATH)
    print(f"\n   模型已儲存至: {MODEL_PATH}")
    print("=" * 55)
    print("   訓練完成！請重啟 app.py 以載入新模型。")
    print("=" * 55)


if __name__ == "__main__":
    train()
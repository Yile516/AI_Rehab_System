# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:24:35 2025

@author: ivanl
"""

import cv2
import pandas as pd
import numpy as np

def draw_info(img, text, pos, color=(0, 255, 0), scale=0.7):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def create_rehab_video(video_path, csv_path, output_path, result_class, features):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # change codec to avc1 (H.264) for web compatibility
    # Fallback logic could be added here, but avc1 is standard for modern opencv/windows
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 顏色定義
    C_OK = (0, 255, 0)
    C_WARN = (0, 165, 255) # 橘色
    C_BAD = (0, 0, 255)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or idx >= len(df): break
        row = df.iloc[idx]
        
        # 1. 取得數據
        trunk = row['trunk_angle']
        dist = row['hand_knee_dist']
        
        # 2. 繪製骨架連線
        # Determine side from 'active_side' column if processing passed it, 
        # but here we rely on the features dict logic or infer from column presence
        # For simplicity, we can infer side key from columns like 'SHOULDER_x' but we need to know index
        # Better: processor.py added 'active_side' to row
        
        side = row['active_side'] if 'active_side' in row else 'right'
        
        kps_names = ['SHOULDER', 'HIP', 'KNEE', 'ANKLE', 'WRIST']
        pts = {}
        for part in kps_names:
            if f'{part}_x' in row and not pd.isna(row[f'{part}_x']):
                pts[part] = (int(row[f'{part}_x']*w), int(row[f'{part}_y']*h))
        
        if len(pts) == 5:
            # 畫身體主軸
            cv2.line(frame, pts['SHOULDER'], pts['HIP'], C_OK, 3)
            cv2.line(frame, pts['HIP'], pts['KNEE'], C_OK, 3)
            cv2.line(frame, pts['KNEE'], pts['ANKLE'], C_OK, 3)
            # 畫手部
            cv2.line(frame, pts['SHOULDER'], pts['WRIST'], C_OK, 2)

            # 3. 代償偵測視覺化 (XAI)
            # 如果偵測到手扶膝蓋 (Label 2) 且當前距離很近 -> 畫紅線警示
            if result_class == 2 and dist < 0.15:
                cv2.line(frame, pts['WRIST'], pts['KNEE'], C_BAD, 5)
                draw_info(frame, "WARNING: Hand Support!", (50, 200), C_BAD, 1.0)
            
            # 如果軀幹前傾過大 -> 身體線條變紅
            if trunk > 50:
                cv2.line(frame, pts['SHOULDER'], pts['HIP'], C_WARN, 5)

        # 4. 儀表板
        cv2.rectangle(frame, (0, h-120), (400, h), (0,0,0), -1)
        draw_info(frame, f"Trunk Lean: {trunk:.1f} deg", (20, h-80), C_OK)
        draw_info(frame, f"Hand-Knee Dist: {dist:.2f}", (20, h-40), C_OK)
        
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    return True
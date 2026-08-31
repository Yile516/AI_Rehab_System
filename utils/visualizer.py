# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:24:35 2025

@author: ivanl
"""

import cv2
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def draw_info(img, text, pos, color=(0, 255, 0), scale=0.7):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def create_rehab_video(video_path, csv_path, output_path, result_class, features):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine if rotation is needed, check features first, fallback to w > h
    best_orient = features.get('best_orientation', 'rotate_90' if w > h else 'raw') if isinstance(features, dict) else ('rotate_90' if w > h else 'raw')
    rotate_needed = (best_orient == 'rotate_90')
    if rotate_needed:
        # Swap width and height for writer
        w, h = h, w
    
    # 確保寫入影片的寬高為 4 的倍數，提高特定 H.264 編碼器的相容性與寫入成功率
    w = (w // 4) * 4
    h = (h // 4) * 4

    # change codec to avc1 (H.264) for web compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        logger.warning(f"avc1 (H.264) 編碼器開啟失敗，嘗試使用 mp4v 作為備用方案。影片解析度: {w}x{h}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 顏色定義
    C_OK = (0, 255, 0)
    C_WARN = (0, 165, 255) # 橘色
    C_BAD = (0, 0, 255)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Apply same rotation as processor
        if rotate_needed:
             frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Ensure we don't go out of bounds if csv is shorter for some reason
        if idx < len(df):
            row = df.iloc[idx]
            
            # Check if row has valid data (e.g. check trunk_angle)
            if pd.notna(row.get('trunk_angle')):
                # 1. 取得數據
                trunk = row['trunk_angle']
                dist = row['hand_knee_dist']
                
                # 2. 繪製骨架連線
                side = row['active_side'] if 'active_side' in row and pd.notna(row['active_side']) else 'right'
                
                kps_names = ['SHOULDER', 'HIP', 'KNEE', 'ANKLE', 'WRIST']
                pts = {}
                for part in kps_names:
                    if f'{part}_x' in row and not pd.isna(row[f'{part}_x']):
                        pts[part] = (int(row[f'{part}_x']*w), int(row[f'{part}_y']*h))
                
                # 畫身體主要結構（若點存在則連接）
                if 'SHOULDER' in pts and 'HIP' in pts:
                    # 若前傾角度大於 50 度，畫橘色警告
                    color = C_WARN if trunk > 50 else C_OK
                    thickness = 5 if trunk > 50 else 3
                    cv2.line(frame, pts['SHOULDER'], pts['HIP'], color, thickness)
                    
                if 'HIP' in pts and 'KNEE' in pts:
                    cv2.line(frame, pts['HIP'], pts['KNEE'], C_OK, 3)
                    
                if 'KNEE' in pts and 'ANKLE' in pts:
                    cv2.line(frame, pts['KNEE'], pts['ANKLE'], C_OK, 3)
                    
                if 'SHOULDER' in pts and 'WRIST' in pts:
                    cv2.line(frame, pts['SHOULDER'], pts['WRIST'], C_OK, 2)

                # 3. 代償偵測視覺化 (XAI)
                if result_class == 2 and pd.notna(dist) and dist < 0.15:
                    if 'WRIST' in pts and 'KNEE' in pts:
                        cv2.line(frame, pts['WRIST'], pts['KNEE'], C_BAD, 5)
                    draw_info(frame, "WARNING: Hand Support!", (50, 200), C_BAD, 1.0)

                # 4. 儀表板
                cv2.rectangle(frame, (0, h-120), (400, h), (0,0,0), -1)
                draw_info(frame, f"Trunk Lean: {trunk:.1f} deg", (20, h-80), C_OK)
                draw_info(frame, f"Hand-Knee Dist: {dist:.2f}", (20, h-40), C_OK)
        
        # 確保寫入的 frame 解析度與 VideoWriter 預期的一致
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    return True
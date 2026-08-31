# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:17:13 2025

@author: ivanl
"""

import os
import cv2
import logging
import mediapipe as mp
import numpy as np
import pandas as pd

# 統一使用 logging 輸出診斷訊息，避免在 Flask WSGI 環境下 print() 引發 OSError
logger = logging.getLogger(__name__)

# 側面觀測關鍵點 (假設拍攝右側)
KP = {
    'SHOULDER': 12, 'HIP': 24, 
    'KNEE': 26, 'ANKLE': 28, 'WRIST': 16 
}

def calculate_angle_3points(a, b, c):
    """計算三點夾角 (一般角度)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def calculate_vertical_angle(p1, p2):
    """計算 p1-p2 連線與垂直線(Y軸)的夾角 (用於軀幹前傾)"""
    v = np.array(p1) - np.array(p2) # 向量
    vertical = np.array([0, -1, 0]) # 垂直向上向量 (MediaPipe Y軸向下，故用-1)
    # 簡化計算：只看 2D 投影 (x, y)
    v_2d = v[:2]
    vert_2d = np.array([0, -1])
    cosine = np.dot(v_2d, vert_2d) / (np.linalg.norm(v_2d) * np.linalg.norm(vert_2d) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def determine_best_orientation(video_path):
    """
    動態檢測影片的最佳朝向：不旋轉 (raw) 或順時針旋轉 90 度 (rotate_90)。
    讀取前 20 幀在兩種朝向下的 MediaPipe Pose 偵測效果來做決策。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 'raw'
        
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    frames_to_check = []
    while len(frames_to_check) < 20:
        ret, frame = cap.read()
        if not ret: break
        frames_to_check.append(frame)
    cap.release()
    
    if not frames_to_check:
        return 'raw'
        
    raw_angles = []
    rot_angles = []
    
    for frame in frames_to_check:
        h, w = frame.shape[:2]
        
        # 1. 測試原始不旋轉
        results_raw = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results_raw.pose_landmarks:
            landmarks = results_raw.pose_landmarks.landmark
            sh = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
            if landmarks[12].visibility > 0.15 and landmarks[24].visibility > 0.15:
                if sh[1] < hip[1]:
                    angle = calculate_vertical_angle(sh, hip)
                    raw_angles.append(angle)
                        
        # 2. 測試順時針旋轉 90 度 (只有寬大於高時才有旋轉的可能)
        if w > h:
            frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            results_rot = pose.process(cv2.cvtColor(frame_rot, cv2.COLOR_BGR2RGB))
            if results_rot.pose_landmarks:
                landmarks = results_rot.pose_landmarks.landmark
                sh = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
                hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
                if landmarks[12].visibility > 0.15 and landmarks[24].visibility > 0.15:
                    if sh[1] < hip[1]:
                        angle = calculate_vertical_angle(sh, hip)
                        rot_angles.append(angle)
                            
    raw_count = len(raw_angles)
    rot_count = len(rot_angles)
    raw_mean = np.mean(raw_angles) if raw_angles else 999
    rot_mean = np.mean(rot_angles) if rot_angles else 999
    
    logger.info(f"影片朝向診斷: {os.path.basename(video_path)} | raw_count={raw_count}, raw_mean={raw_mean:.1f}° | rot_count={rot_count}, rot_mean={rot_mean:.1f}°")
    
    if raw_count == 0 and rot_count == 0:
        return 'raw'
    elif raw_count > 0 and rot_count == 0:
        return 'raw'
    elif rot_count > 0 and raw_count == 0:
        return 'rotate_90'
    else:
        if raw_mean < 75.0 and rot_mean >= 75.0:
            return 'raw'
        elif rot_mean < 75.0 and raw_mean >= 75.0:
            return 'rotate_90'
        else:
            return 'raw' if raw_mean <= rot_mean else 'rotate_90'

def process_video(video_path, output_csv):
    best_orientation = determine_best_orientation(video_path)
    logger.info(f"開始處理影片 {video_path}，使用自適應朝向: {best_orientation}")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    frame_count = 0
    raw_frames_data = []

    # 第一階段：快速偵測所有影格的 Pose 關節點，儲存至記憶體
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        if best_orientation == 'rotate_90' and w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        frame_data = {
            'frame': frame_count,
            'time': frame_count / fps,
            'landmarks': None
        }
        
        if results.pose_landmarks:
            frame_data['landmarks'] = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
            
        raw_frames_data.append(frame_data)
        frame_count += 1
        
    cap.release()
    
    if not raw_frames_data:
        return None, None

    # 第二階段：統計整部影片中，左側與右側關節點的有效偵測影格數，決定最佳觀測側面
    left_valid_count = 0
    right_valid_count = 0
    
    for fd in raw_frames_data:
        lms = fd['landmarks']
        if lms:
            # 檢查左側必備關鍵點 (11, 23, 25, 27)
            if all(lms[i]['visibility'] >= 0.15 for i in [11, 23, 25, 27]):
                left_valid_count += 1
            # 檢查右側必備關鍵點 (12, 24, 26, 28)
            if all(lms[i]['visibility'] >= 0.15 for i in [12, 24, 26, 28]):
                right_valid_count += 1
                
    chosen_side = 'left' if left_valid_count >= right_valid_count else 'right'
    logger.info(f"整部影片觀測側面統計：左側有效影格數={left_valid_count}, 右側有效影格數={right_valid_count}。最終決定採用：{chosen_side}")

    # 第三階段：依照決定的觀測側面，進行資料萃取、平滑化(EMA)與特徵計算
    data = []
    ema_alpha = 0.4
    smoothed_kps = {}
    
    if chosen_side == 'left':
        kp_map = {'SHOULDER': 11, 'HIP': 23, 'KNEE': 25, 'ANKLE': 27, 'WRIST': 15}
    else:
        kp_map = {'SHOULDER': 12, 'HIP': 24, 'KNEE': 26, 'ANKLE': 28, 'WRIST': 16}

    for fd in raw_frames_data:
        row = {'frame': fd['frame'], 'time': fd['time']}
        lms = fd['landmarks']
        
        if lms:
            valid_frame = True
            for name in ['SHOULDER', 'HIP', 'KNEE', 'ANKLE']:
                idx = kp_map[name]
                if lms[idx]['visibility'] < 0.15:
                    valid_frame = False
                    break
                row[f'{name}_x'] = lms[idx]['x']
                row[f'{name}_y'] = lms[idx]['y']
                row[f'{name}_z'] = lms[idx]['z']
                
            if valid_frame:
                wrist_idx = kp_map['WRIST']
                if lms[wrist_idx]['visibility'] >= 0.15:
                    row['WRIST_x'] = lms[wrist_idx]['x']
                    row['WRIST_y'] = lms[wrist_idx]['y']
                    row['WRIST_z'] = lms[wrist_idx]['z']
                    wrist = [lms[wrist_idx]['x'], lms[wrist_idx]['y'], lms[wrist_idx]['z']]
                else:
                    row['WRIST_x'] = np.nan
                    row['WRIST_y'] = np.nan
                    row['WRIST_z'] = np.nan
                    wrist = [np.nan, np.nan, np.nan]

                shoulder = [row['SHOULDER_x'], row['SHOULDER_y'], row['SHOULDER_z']]
                hip = [row['HIP_x'], row['HIP_y'], row['HIP_z']]
                knee = [row['KNEE_x'], row['KNEE_y'], row['KNEE_z']]
                ankle = [row['ANKLE_x'], row['ANKLE_y'], row['ANKLE_z']]

                # 套用 EMA 濾波器以克服骨架點抖動
                current_kps = {
                    'SHOULDER': shoulder,
                    'HIP': hip,
                    'KNEE': knee,
                    'ANKLE': ankle
                }
                if not np.any(np.isnan(wrist)):
                    current_kps['WRIST'] = wrist
                    
                for k, v in current_kps.items():
                    if k not in smoothed_kps:
                        smoothed_kps[k] = np.array(v)
                    else:
                        smoothed_kps[k] = ema_alpha * np.array(v) + (1 - ema_alpha) * smoothed_kps[k]
                
                shoulder = smoothed_kps['SHOULDER']
                hip = smoothed_kps['HIP']
                knee = smoothed_kps['KNEE']
                ankle = smoothed_kps['ANKLE']
                
                if 'WRIST' in smoothed_kps and not np.any(np.isnan(wrist)):
                    wrist = smoothed_kps['WRIST']
                else:
                    wrist = np.array([np.nan, np.nan, np.nan])
                
                # 更新回 row，讓輸出的 CSV 也是平滑過的數據
                row['SHOULDER_x'], row['SHOULDER_y'], row['SHOULDER_z'] = shoulder
                row['HIP_x'], row['HIP_y'], row['HIP_z'] = hip
                row['KNEE_x'], row['KNEE_y'], row['KNEE_z'] = knee
                row['ANKLE_x'], row['ANKLE_y'], row['ANKLE_z'] = ankle
                if not np.isnan(wrist[0]):
                    row['WRIST_x'], row['WRIST_y'], row['WRIST_z'] = wrist
                else:
                    row['WRIST_x'], row['WRIST_y'], row['WRIST_z'] = np.nan, np.nan, np.nan

                # 3. Calculate Features
                try:
                    row['trunk_angle'] = calculate_vertical_angle(shoulder, hip)
                    row['knee_angle'] = calculate_angle_3points(hip, knee, ankle)
                    
                    # Hand-Knee Dist
                    if not np.isnan(wrist[0]):
                        w_pt, k_pt = np.array(wrist[:2]), np.array(knee[:2])
                        row['hand_knee_dist'] = np.linalg.norm(w_pt - k_pt)
                    else:
                        row['hand_knee_dist'] = np.nan
                    
                    # 物理安全過濾：前傾角度 > 80° 屬於不合理的人體 STS 姿勢（BlazePose 常將椅背誤判為人體）
                    # 判定此幀為誤判，清空特徵與座標，避免骨架飄移至椅子上
                    if row['trunk_angle'] > 80.0:
                        valid_frame = False
                        for name in kp_map.keys():
                            row.pop(f'{name}_x', None)
                            row.pop(f'{name}_y', None)
                            row.pop(f'{name}_z', None)
                        row.pop('trunk_angle', None)
                        row.pop('knee_angle', None)
                        row.pop('hand_knee_dist', None)
                except Exception as e:
                    logger.warning(f"Calculation error at frame {fd['frame']}: {e}")
                    valid_frame = False

                row['active_side'] = chosen_side

        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Needs at least some valid data to be useful, but for sync we return what we have
    if df.empty: return None, None
    
    # 檢查是否含有足夠的特徵欄位，避免後續計算時間注意力機制或特徵提取時引發 KeyError
    required_cols = ['trunk_angle', 'knee_angle', 'hand_knee_dist']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols or df['trunk_angle'].dropna().empty:
        logger.warning(f"影片中未偵測到足夠的有效骨架特徵。缺失欄位: {missing_cols}")
        df.to_csv(output_csv, index=False)
        return df, None
    
    # 進行線性插補填補缺失的骨架點與特徵值，限制最多連續插補 3 幀以防範大範圍漏檢時骨架飄移跑掉
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction='both', limit=3)
    
    # 若 hand_knee_dist 長期缺失（例如雙手抱胸），填補安全值 0.40 表示無代償
    if 'hand_knee_dist' in df.columns:
        df['hand_knee_dist'] = df['hand_knee_dist'].fillna(0.40)
        
    df.to_csv(output_csv, index=False)
    
    # 引入時間注意力機制 (Temporal Attention Mechanism)
    # 利用前傾角度的變化速度 (Velocity) 作為注意力權重，抓出對應動作最劇烈（最關鍵）的瞬間
    if len(df) > 1:
        df['trunk_vel'] = df['trunk_angle'].diff().abs().fillna(0)
        # Softmax 權重計算
        max_vel = df['trunk_vel'].max()
        if max_vel > 0:
            exp_w = np.exp(df['trunk_vel'] - max_vel)
            df['attention_weight'] = exp_w / exp_w.sum()
        else:
            df['attention_weight'] = 1.0 / len(df)
            
        # 依據評審意見修正特徵提取邏輯：
        # X1_MaxTrunkLean：改用全域最大軀幹前傾角，避開角速度峰值幀非最大前傾角之問題。
        att_trunk_lean = df['trunk_angle'].max()
        
        # X4_EndKneeAngle：改用動作結尾穩定起立期之平均膝蓋角度，避開起立劇烈時的半蹲角度。
        # 為了提升對最後影片片段可能缺失的容錯性，改取最後 10 個有效偵測值 (non-NaN) 的平均值。
        valid_knees = df['knee_angle'].dropna() if 'knee_angle' in df.columns else pd.Series()
        if len(valid_knees) >= 10:
            att_knee_angle = valid_knees.iloc[-10:].mean()
        elif len(valid_knees) > 0:
            att_knee_angle = valid_knees.mean()
        else:
            att_knee_angle = 0.0
            
        # X3_AttHandKneeDist：採用注意力加權手膝距離，使代償最明顯的起身瞬間獲得更高權重。
        att_hand_knee_dist = (df['hand_knee_dist'] * df['attention_weight']).sum() 
    else:
        att_trunk_lean = df['trunk_angle'].max()
        att_knee_angle = df['knee_angle'].max()
        att_hand_knee_dist = df['hand_knee_dist'].min()

    # --- Summary Statistics ---
    features = {
        'X1_MaxTrunkLean': att_trunk_lean,
        'X2_TotalDuration': df['time'].max(),
        'X3_AttHandKneeDist': att_hand_knee_dist,
        'X4_EndKneeAngle': att_knee_angle,
        'DetectedSide': chosen_side,
        'best_orientation': best_orientation
    }
    
    # Handle NaN
    for k in features:
        if k != 'DetectedSide' and pd.isna(features[k]): features[k] = 0.0
        
    return df, features
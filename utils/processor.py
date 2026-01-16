# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:17:13 2025

@author: ivanl
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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

def process_video(video_path, output_csv):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    data = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30

    frame_count = 0
    # Add buffer to determine side from first few frames
    side_buffer = [] 
    determined_side = None # 'right' or 'left'

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Auto-rotate if landscape but likely portrait content
        # Simple heuristic: If width > height, rotate 90 degrees clockwise (or counter-clockwise depending on metadata, but assuming standard phone issues)
        # For robustness, we can try to rely on MediaPipe, but simpler is better:
        # If width > height -> Rotate to make it height > width (Portrait)
        h, w = frame.shape[:2]
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        row = {'frame': frame_count, 'time': frame_count/fps}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. Side Detection (Robust with buffer)
            if determined_side is None:
                left_vis = landmarks[11].visibility + landmarks[23].visibility
                right_vis = landmarks[12].visibility + landmarks[24].visibility
                
                # Check absolute visibility to avoid noise
                if max(left_vis, right_vis) > 1.0: # At least one side decent
                    side_buffer.append('left' if left_vis > right_vis else 'right')
                
                # Wait for buffer to fill or default if taking too long
                if len(side_buffer) >= 30: # Use Config value via import if possible, hardcoded 30 for now to match plan
                    determined_side = max(set(side_buffer), key=side_buffer.count)
                    print(f"Analyzed Side: {determined_side}")
                elif frame_count > 60: # Timeout case
                     print("Side detection timed out, defaulting to right")
                     determined_side = 'right'

            # Define KP Map based on side
            # Only process if side is determined or we fallback
            current_side = determined_side if determined_side else 'right'
            
            if current_side == 'left':
                kp_map = {'SHOULDER': 11, 'HIP': 23, 'KNEE': 25, 'ANKLE': 27, 'WRIST': 15}
            else: # Default to right
                kp_map = {'SHOULDER': 12, 'HIP': 24, 'KNEE': 26, 'ANKLE': 28, 'WRIST': 16}

            # 2. Extract Data
            valid_frame = True
            for name, idx in kp_map.items():
                lm = landmarks[idx]
                if lm.visibility < 0.3:
                    valid_frame = False
                    break
                row[f'{name}_x'], row[f'{name}_y'], row[f'{name}_z'] = lm.x, lm.y, lm.z
            
            if valid_frame:
                shoulder = [row['SHOULDER_x'], row['SHOULDER_y'], row['SHOULDER_z']]
                hip = [row['HIP_x'], row['HIP_y'], row['HIP_z']]
                knee = [row['KNEE_x'], row['KNEE_y'], row['KNEE_z']]
                ankle = [row['ANKLE_x'], row['ANKLE_y'], row['ANKLE_z']]
                wrist = [row['WRIST_x'], row['WRIST_y'], row['WRIST_z']]

                # 3. Calculate Features
                try:
                    row['trunk_angle'] = calculate_vertical_angle(shoulder, hip)
                    row['knee_angle'] = calculate_angle_3points(hip, knee, ankle)
                    
                    # Hand-Knee Dist
                    w_pt, k_pt = np.array(wrist[:2]), np.array(knee[:2])
                    row['hand_knee_dist'] = np.linalg.norm(w_pt - k_pt)
                except Exception as e:
                    print(f"Calculation error at frame {frame_count}: {e}")
                    valid_frame = False

                row['active_side'] = current_side

        data.append(row)
        frame_count += 1
        
    cap.release()
    df = pd.DataFrame(data)
    
    # Needs at least some valid data to be useful, but for sync we return what we have
    if df.empty: return None, None
    
    df.to_csv(output_csv, index=False)
    
    # --- Summary Statistics ---
    features = {
        'X1_MaxTrunkLean': df['trunk_angle'].max(),
        'X2_TotalDuration': df['time'].max(),
        'X3_MinHandKneeDist': df['hand_knee_dist'].min(),
        'X4_EndKneeAngle': df['knee_angle'].max(),
        'DetectedSide': df['active_side'].iloc[0] if 'active_side' in df else 'right'
    }
    
    # Handle NaN
    for k in features:
        if k != 'DetectedSide' and pd.isna(features[k]): features[k] = 0
        
    return df, features
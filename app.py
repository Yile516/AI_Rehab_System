# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:25:08 2025

@author: ivanl
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from utils.processor import process_video
from utils.visualizer import create_rehab_video

app = Flask(__name__)
app.config.from_object(Config)

# 載入模型
try:
    model = joblib.load(Config.MODEL_PATH)
    print("AI 模型載入成功")
except:
    print("模型未找到，請先執行 model/train_model.py")
    model = None

RESULT_MAP = {
    0: {"status": "健康 (Healthy)", "desc": "動作標準，下肢肌力良好。", "color": "green"},
    1: {"status": "衰弱風險 (Frailty)", "desc": "軀幹前傾過大或動作緩慢，建議加強核心與大腿肌力。", "color": "orange"},
    2: {"status": "代償動作 (Compensation)", "desc": "偵測到手扶膝蓋借力！這是肌力不足的警訊，請避免此習慣。", "color": "red"}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('video')
        if file:
            return process_and_analyze(file)
            
    return render_template('index.html')

@app.route('/record')
def record_page():
    return render_template('record.html')

@app.route('/analyze_blob', methods=['POST'])
def analyze_blob():
    file = request.files.get('video')
    if file:
        return process_and_analyze(file)
    return "No video uploaded", 400

def process_and_analyze(file):
    # 1. 儲存影片
    filename = secure_filename(file.filename)
    if not filename: filename = "upload_video.mp4"
    vid_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(vid_path)
    
    # 2. 分析特徵
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], filename + '.csv')
    df, features = process_video(vid_path, csv_path)
    
    if features:
        # 3. AI 預測
        # 轉換成 DataFrame 輸入模型 (注意順序要跟訓練時一樣)
        X_input = pd.DataFrame([features])[['X1_MaxTrunkLean', 'X2_TotalDuration', 'X3_MinHandKneeDist', 'X4_EndKneeAngle']]
        pred = model.predict(X_input)[0]
        res_info = RESULT_MAP[pred]
        
        # 4. 生成結果影片
        out_vid_name = 'result_' + filename
        out_vid_path = os.path.join(app.config['RESULTS_FOLDER'], out_vid_name)
        create_rehab_video(vid_path, csv_path, out_vid_path, pred, features)
        
        return render_template('result.html', info=res_info, features=features, video=out_vid_name)
    else:
        return "Analysis Failed or Video too short"

@app.route('/results/<filename>')
def get_result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
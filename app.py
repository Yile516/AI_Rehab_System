# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:25:08 2025

@author: ivanl
"""

import os
import logging
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from utils.processor import process_video
from utils.visualizer import create_rehab_video

# 設定 logging，訊息輸出至 stderr（Flask 標準做法，不受 WSGI stdout 影響）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# 載入模型
try:
    model = joblib.load(Config.MODEL_PATH)
    logger.info("AI 模型載入成功")
except Exception as e:
    logger.warning(f"模型未找到，請先執行 model/train_model.py: {e}")
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
    import time
    # 1. 儲存影片，保留安全檔名與副檔名
    orig_name, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    if ext not in ['.mp4', '.mov', '.avi', '.webm']:
        ext = '.mp4'
        
    sec_name = secure_filename(orig_name)
    if not sec_name:
        sec_name = f"upload_{int(time.time())}"
        
    filename = sec_name + ext
    vid_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(vid_path)
    
    # 2. 分析特徵
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], filename + '.csv')
    df, features = process_video(vid_path, csv_path)
    
    if features:
        # 3. AI 預測
        # 轉換成 DataFrame 輸入模型 (注意順序要跟訓練時一樣)
        X_input = pd.DataFrame([features])[['X1_MaxTrunkLean', 'X2_TotalDuration', 'X3_AttHandKneeDist', 'X4_EndKneeAngle']]
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
    app.run(host='0.0.0.0', port=5000, debug=True)
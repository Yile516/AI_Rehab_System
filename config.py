# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:17:12 2025

@author: ivanl
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'rehab_model.pkl')
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}

    # 確保資料夾存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
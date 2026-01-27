🦾 AI Rehab System (AI 復健系統)中文介紹 | English Description📸 Demo Showcase<a name="繁體中文"></a>🇹🇼 繁體中文說明這是一套結合 電腦視覺 (MediaPipe) 與 機器學習 (Scikit-learn) 的智慧復健系統，專為老年人「從坐到站 (Sit-to-Stand)」動作設計。系統能即時偵測骨架、計算關節角度，並透過 AI 模型分析動作階段，提供即時回饋。🚀 如何在其他電腦執行？準備環境：安裝 Python 3.9+ (記得勾選 Add to PATH) 與 Git。下載專案：Bashgit clone https://github.com/Yile516/AI_Rehab_System.git
cd AI_Rehab_System
安裝套件：Bashpip install -r requirements.txt
啟動系統：Bashpython app.py
打開瀏覽器輸入 http://127.0.0.1:5000 即可開始。<a name="english"></a>🇺🇸 English DescriptionAI Rehab System is an intelligent solution designed for elderly Sit-to-Stand (STS) kinematics analysis. By integrating MediaPipe for pose estimation and Scikit-learn for movement classification, it provides a low-cost, professional-grade rehabilitation tool for home use.🛠️ Key FeaturesReal-time Biometrics: Tracks 33 body landmarks and calculates knee/hip angles.AI Classification: Automatically detects movement phases (Sitting, Rising, Standing).Web-based UI: Easy-to-use interface built with Flask.💻 Setup & ExecutionClone the Repo:Bashgit clone https://github.com/Yile516/AI_Rehab_System.git
cd AI_Rehab_System
Install Dependencies:Bashpip install -r requirements.txt
Run App:Bashpython app.py
📂 專案結構與 Git 指南 (Development Guide)🔄 Git Cheat Sheet (開發者必備)目標 (Goal)指令 (Command)說明 (Description)獲取更新git pull開始工作前，先抓取雲端最新進度。收集變更git add .修改完 Code 後，把檔案放進暫存區。確認存檔git commit -m "update"為這次修改寫下備註（版本存檔）。推送雲端git push將存檔點正式上傳到 GitHub。🏗️ 工作流程圖 (Workflow)程式碼片段graph TD
    A[Start Working] --> B[git pull]
    B --> C[Coding & Testing]
    C --> D[git add .]
    D --> E[git commit -m '...']
    E --> F[git push]
    F --> G((GitHub Cloud Success))
📈 系統截圖 (System Screenshots)<p align="center"><img src="https://github.com/user-attachments/assets/40579879-8c96-4e48-8e63-775de8b09563" width="30%" /><img src="https://github.com/user-attachments/assets/96657213-eb2c-4112-b4d5-ab8a671ed5c3" width="30%" /><img src="https://github.com/user-attachments/assets/c3bf6cb9-276b-48da-8110-4e6101eb4616" width="30%" /></p>

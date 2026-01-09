# AI Rehab System (AI 復健系統)
這是你的 AI 復健系統專案。以下說明如何在其他電腦上下載並執行這個程式。
## 1. 在新電腦上的準備工作
在開始之前，請確保新電腦已經安裝：
1.  **Python** (建議 3.9 或以上版本): [下載 Python](https://www.python.org/downloads/)
    *   *安裝時記得勾選 "Add Python to PATH"*
2.  **Git**: [下載 Git](https://git-scm.com/downloads)
## 2. 下載專案 (Clone)
打開終端機 (CMD 或 PowerShell)，執行以下指令將專案下載到那台電腦：
```bash
git clone https://github.com/Yile516/AI_Rehab_System.git
cd AI_Rehab_System
```
## 3. 安裝套件
下載完後，需要安裝專案所需的 Python 函式庫。在專案資料夾內執行：
```bash
pip install -r requirements.txt
```
## 4. 執行程式
安裝完成後，就可以啟動網頁伺服器：
```bash
python app.py
```
看到 `Running on http://127.0.0.1:5000` 表示啟動成功，打開瀏覽器輸入該網址即可使用。

@echo off
echo [*] the environment is being prepared...
python -m venv venv
call venv\Scripts\activate
echo [*] Installing necessary packages...
pip install --upgrade pip
pip install -r requirements.txt
echo [✓] installation completed.

echo [*] application is starting...
python send_coord.py

pause

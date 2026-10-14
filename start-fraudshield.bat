@echo off
setlocal
cd /d "%~dp0"
title FraudShield launcher

where python >nul 2>&1 || (echo [x] Python was not found in PATH. & pause & exit /b 1)

python -c "import fastapi, uvicorn, xgboost, shap, river, imblearn, sklearn" >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies, first run only...
  python -m pip install -r requirements.txt || (echo [x] pip install failed. & pause & exit /b 1)
)

if not exist artifacts\model.joblib (
  if not exist data\creditcard.csv (
    echo [x] No trained model and no data\creditcard.csv.
    echo     Put the Kaggle CSV in the data folder, then run this file again.
    pause
    exit /b 1
  )
  echo Training the model, this takes a few minutes...
  python train.py || (echo [x] Training failed. & pause & exit /b 1)
)

curl -sf -m 2 http://127.0.0.1:8000/health >nul 2>&1
if not errorlevel 1 goto api_running

echo.
choice /c FK /t 6 /d F /m "Fresh demo data (F, default after 6 s) or keep history (K)"
if errorlevel 2 goto keep
del /q artifacts\fraudshield.db artifacts\feedback.jsonl >nul 2>&1
echo Demo data cleared: transactions, labels and merchant reports start at 0.
:keep
echo Starting the API on port 8000...
start "FraudShield API" cmd /k python -m uvicorn app:app --host 127.0.0.1 --port 8000
goto init_done

:api_running
echo API is already running, so its data was kept.
echo Run stop-fraudshield.bat first if you want a fresh start.

:init_done
set /a tries=0
:wait
curl -sf -m 2 http://127.0.0.1:8000/health >nul 2>&1 && goto up
set /a tries+=1
if %tries% geq 60 (
  echo [x] The API did not start within 60 seconds. Check the FraudShield API window.
  pause
  exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait

:up
curl -sf -m 2 http://127.0.0.1:5500/checkout.html >nul 2>&1
if errorlevel 1 (
  echo Starting the demo checkout page on port 5500...
  start "FraudShield demo page" cmd /k python -m http.server 5500 --bind 127.0.0.1 --directory demo
  ping -n 3 127.0.0.1 >nul
)

start "" http://127.0.0.1:8000
start "" http://127.0.0.1:5500/checkout.html

echo.
echo FraudShield is running.
echo   Dashboard:  http://127.0.0.1:8000
echo   Demo page:  http://127.0.0.1:5500/checkout.html
echo Run stop-fraudshield.bat to shut everything down.
ping -n 6 127.0.0.1 >nul
endlocal

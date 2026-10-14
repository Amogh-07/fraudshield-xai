@echo off
for %%P in (8000 5500) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%P " ^| findstr LISTENING') do (
    taskkill /PID %%a /T /F >nul 2>&1
  )
)
taskkill /FI "WINDOWTITLE eq FraudShield*" /F >nul 2>&1
echo FraudShield stopped.
ping -n 3 127.0.0.1 >nul

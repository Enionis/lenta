@echo off
chcp 65001 >nul
echo ==========================================
echo Lenta Tech - Price Tag Recognition ML
echo ==========================================
echo.

REM Активация виртуального окружения
call .\venv\Scripts\activate.bat

REM Запуск CLI с параметрами
python -m ml.cli %*

REM Деактивация
call deactivate

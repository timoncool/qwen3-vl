@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo Запуск Qwen3-VL Image Description Generator
echo ========================================
echo.

REM Определяем директорию скрипта
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Проверяем наличие Python
if not exist "python\python.exe" (
    echo ОШИБКА: Python не найден!
    echo.
    echo Пожалуйста, сначала запустите install.bat для установки.
    echo.
    pause
    exit /b 1
)

REM Проверяем наличие app.py
if not exist "app.py" (
    echo ОШИБКА: Файл app.py не найден!
    echo.
    pause
    exit /b 1
)

REM Читаем версию CUDA из конфигурационного файла
if exist "cuda_version.txt" (
    set /p CUDA_VERSION=<cuda_version.txt
    echo Используется: !CUDA_VERSION!
    echo.
)

REM Устанавливаем переменные окружения для оптимизации
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

REM Запускаем приложение
echo Запуск приложения...
echo.
echo После запуска приложение будет доступно по адресу:
echo http://localhost:7860
echo.
echo Для остановки приложения нажмите Ctrl+C
echo.
echo ========================================
echo.

python\python.exe app.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ОШИБКА при запуске приложения!
    echo ========================================
    echo.
    echo Возможные причины:
    echo 1. Не установлены зависимости - запустите install.bat
    echo 2. Недостаточно памяти GPU/RAM
    echo 3. Проблемы с CUDA драйверами
    echo.
    pause
    exit /b 1
)

pause

@echo off
setlocal

set PYTHON=python

%PYTHON% -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.10+ first.
  exit /b 1
)

if not exist .venv (
  %PYTHON% -m venv .venv
)

call .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo NOTE: For GPU training on NVIDIA, install a CUDA-enabled PyTorch build:
echo   https://pytorch.org/get-started/locally/
echo.

python app.py

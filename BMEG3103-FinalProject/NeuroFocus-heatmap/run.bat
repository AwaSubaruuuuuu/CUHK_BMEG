@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set PY=py -3
%PY% -c "import sys" 2>nul || (
  echo [x] 找不到 Python 3，请先安装 https://www.python.org/downloads/ 并勾选 Add to PATH
  pause
  exit /b 1
)

%PY% -c "import numpy" 2>nul
if errorlevel 1 (
  echo [!] 缺少依赖，正在安装 numpy / pyserial ...
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [x] 安装失败，请手动执行: %PY% -m pip install numpy pyserial
    pause
    exit /b 1
  )
)

echo [i] 启动本地推理服务 http://127.0.0.1:8765 ...
echo     默认用模拟数据（演示用）。真头环请改用: run.bat --source serial --serial-port COM7
echo     保持这个窗口开着，然后到 Chrome 里点 B 站视频页右下角的悬浮按钮。
echo.
%PY% server.py %*
pause

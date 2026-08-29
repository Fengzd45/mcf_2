@echo off
chcp 65001 >nul
echo 双向实时语音互译 — 启动脚本

:: ── 检查环境变量 ──────────────────────────────────────────
if "%DASHSCOPE_API_KEY%"=="" (
  echo ❌ 请先设置 DASHSCOPE_API_KEY
  echo    set DASHSCOPE_API_KEY=sk-xxx
  pause
  exit /b 1
)

if "%DASHSCOPE_WORKSPACE_ID%"=="" (
  echo ❌ 请先设置 DASHSCOPE_WORKSPACE_ID
  echo    set DASHSCOPE_WORKSPACE_ID=ws-xxx
  pause
  exit /b 1
)

:: ── 可选参数默认值 ────────────────────────────────────────
if "%BAILIAN_REGION%"==""         set BAILIAN_REGION=cn-beijing
if "%LIVETRANSLATE_MODEL%"==""    set LIVETRANSLATE_MODEL=qwen3.5-livetranslate-flash-realtime
if "%HOST%"==""                   set HOST=0.0.0.0
if "%PORT%"==""                   set PORT=8000

echo ✅ API Key: %DASHSCOPE_API_KEY:~0,8%...
echo ✅ Workspace: %DASHSCOPE_WORKSPACE_ID%
echo ✅ Region: %BAILIAN_REGION%
echo ✅ Model: %LIVETRANSLATE_MODEL%
echo.

:: ── 安装依赖 ──────────────────────────────────────────────
echo 📦 检查依赖...
pip install -r requirements.txt --quiet

:: ── 启动服务 ──────────────────────────────────────────────
echo 🚀 启动服务: http://%HOST%:%PORT%
echo 按 Ctrl+C 停止
echo.

uvicorn main:app --host %HOST% --port %PORT% --reload
pause
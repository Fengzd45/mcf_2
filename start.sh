#!/bin/bash
# 双向实时语音互译 — 启动脚本

set -e

# ── 检查环境变量 ──────────────────────────────────────────
if [ -z "$DASHSCOPE_API_KEY" ]; then
  echo "❌ 请先设置 DASHSCOPE_API_KEY"
  echo "   export DASHSCOPE_API_KEY=sk-xxx"
  exit 1
fi

if [ -z "$DASHSCOPE_WORKSPACE_ID" ]; then
  echo "❌ 请先设置 DASHSCOPE_WORKSPACE_ID"
  echo "   export DASHSCOPE_WORKSPACE_ID=ws-xxx"
  exit 1
fi

# ── 可选参数默认值 ────────────────────────────────────────
export BAILIAN_REGION="${BAILIAN_REGION:-cn-beijing}"
export LIVETRANSLATE_MODEL="${LIVETRANSLATE_MODEL:-qwen3.5-livetranslate-flash-realtime}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

echo "✅ API Key: ${DASHSCOPE_API_KEY:0:8}..."
echo "✅ Workspace: $DASHSCOPE_WORKSPACE_ID"
echo "✅ Region: $BAILIAN_REGION"
echo "✅ Model: $LIVETRANSLATE_MODEL"
echo ""

# ── 安装依赖 ──────────────────────────────────────────────
if ! python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null; then
  echo "📦 安装依赖..."
  pip install -r requirements.txt --quiet
fi

# ── 启动服务 ──────────────────────────────────────────────
echo "🚀 启动服务: http://${HOST}:${PORT}"
echo "   手机/平板访问本机IP: http://$(hostname -I | awk '{print $1}'):${PORT}"
echo ""
echo "按 Ctrl+C 停止"
echo ""

uvicorn main:app --host "$HOST" --port "$PORT" --reload
import os
import json
import time
import base64
import asyncio
import logging
import requests
from typing import Dict, Optional
from fastapi import FastAPI, WebSocketDisconnect, WebSocket
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 环境变量 ──────────────────────────────────────────────
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

# ── 模型配置 ──────────────────────────────────────────────
ASR_MODEL = "fun-asr-realtime"
TRANSLATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/machine-translation/translation"
TRANSLATE_MODEL = "qwen-mt-turbo"
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "longxiaochun_v2"

# ── 语言映射 ──────────────────────────────────────────────
LANG_MAP = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es", "ru": "ru"
}

rooms: Dict[str, Dict] = {}

# ── 音频看门狗配置 ─────────────────────────────────────────
WATCHDOG_CHECK_INTERVAL = 5
WATCHDOG_IDLE_TIMEOUT = 15


# ── ASR 回调 ──────────────────────────────────────────────
class StreamingCallback(RecognitionCallback):
    def __init__(self, client_id: str, loop: asyncio.AbstractEventLoop, room_id: str):
        self.client_id = client_id
        self.loop = loop
        self.room_id = room_id
        self.is_broken = False

    def on_open(self):
        logger.info(f"✅ ASR 流式会话已建立: {self.client_id}")

    def on_close(self):
        self.is_broken = True
        logger.info(f"ASR 流式会话已关闭: {self.client_id}")

    def on_complete(self):
        logger.info(f"ASR 流式会话正常结束: {self.client_id}")

    def on_error(self, result):
        self.is_broken = True
        error_msg = "未知错误"
        try:
            error_msg = str(result) if result else "未知错误"
            if hasattr(result, 'status_code'):
                error_msg = f"status_code={result.status_code}, {error_msg}"
            if hasattr(result, 'message'):
                error_msg = f"message={result.message}, {error_msg}"
        except Exception as e:
            error_msg = f"无法解析错误详情: {e}"
        logger.error(f"❌ ASR 错误: {error_msg}")

        try:
            asyncio.run_coroutine_threadsafe(
                self._notify_error(error_msg),
                self.loop
            )
        except Exception as e:
            logger.error(f"通知前端 ASR 错误失败: {e}")

    async def _notify_error(self, msg):
        room_id = self.room_id
        client_id = self.client_id
        if room_id in rooms and client_id in rooms[room_id]["clients"]:
            ws = rooms[room_id]["clients"][client_id]
            try:
                await ws.send_text(json.dumps({
                    "type": "asr_error",
                    "msg": f"ASR 错误: {msg}"
                }))
            except Exception:
                pass

    def on_event(self, result):
        try:
            sentence = result.get_sentence()
        except Exception as e:
            logger.error(f"解析识别结果失败: {e}")
            return
        if not sentence:
            return
        if isinstance(sentence, list):
            for s in sentence:
                self._handle_sentence(s)
        else:
            self._handle_sentence(sentence)

    def _handle_sentence(self, sentence):
        if isinstance(sentence, dict):
            text = (sentence.get("text") or "").strip()
            is_end = bool(sentence.get("sentence_end", False))
        else:
            text = str(getattr(sentence, "text", "")).strip()
            is_end = bool(getattr(sentence, "sentence_end", False))
        if not text:
            return

        if is_end:
            logger.info(f"📝 ASR 断句完成 [{self.client_id}]: '{text}'")
        else:
            logger.info(f"📝 ASR 中间结果 [{self.client_id}]: '{text}'")

        asyncio.run_coroutine_threadsafe(
            handle_asr_result(self.client_id, text, self.room_id, is_end),
            self.loop
        )


# ── 翻译（阻塞调用） ──────────────────────────────────────
def _translate_text_blocking(text: str, target_lang: str) -> str:
    if not text or not DASHSCOPE_API_KEY:
        return text
    target = LANG_MAP.get(target_lang, "en")
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": TRANSLATE_MODEL,
        "input": {"text": text, "source_lang": "auto", "target_lang": target}
    }
    try:
        resp = requests.post(TRANSLATE_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json().get("output", {}).get("text", text)
            logger.info(f"🌐 翻译结果 ({target_lang}): {result}")
            return result
        logger.error(f"翻译请求失败: status={resp.status_code}, body={resp.text[:300]}")
        return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text


async def translate_text(text: str, target_lang: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _translate_text_blocking, text, target_lang)


# ── TTS（阻塞调用） ──────────────────────────────────────
def _synthesize_speech_blocking(text: str) -> Optional[bytes]:
    if not text or not DASHSCOPE_API_KEY:
        return None
    try:
        synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE)
        audio = synthesizer.call(text)
        logger.info(f"🔊 TTS 合成成功: {text[:30]}...")
        return audio
    except Exception as e:
        logger.error(f"TTS 失败: {e}")
        return None


async def synthesize_speech(text: str) -> Optional[bytes]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _synthesize_speech_blocking, text)


# ── FastAPI ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 同声传译服务器启动（阿里百炼 SDK 版）")
    yield
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)


# ── 静态文件 ──────────────────────────────────────────────
# 完全不用 StaticFiles，改用路由直接返回文件内容

@app.get("/")
async def get_index():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/{filename}")
async def get_static_file(filename: str):
    """返回 static 目录下的其他文件（CSS、JS 等）"""
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 根据文件扩展名设置 MIME 类型
        if filename.endswith(".css"):
            return HTMLResponse(content=content, media_type="text/css")
        elif filename.endswith(".js"):
            return HTMLResponse(content=content, media_type="application/javascript")
        elif filename.endswith(".html"):
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(content=content)
    return HTMLResponse(content="File not found", status_code=404)


# ── WebSocket 端点 ────────────────────────────────────────
@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, client_id: str):
    await websocket.accept()
    logger.info(f"✅ 客户端 {client_id} 加入房间 {room_id}")

    if room_id not in rooms:
        rooms[room_id] = {"clients": {}, "languages": {}}
    rooms[room_id]["clients"][client_id] = websocket
    await broadcast_room_status(room_id)

    loop = asyncio.get_running_loop()
    callback = StreamingCallback(client_id, loop, room_id)

    try:
        recognition = Recognition(
            model=ASR_MODEL,
            format="pcm",
            sample_rate=16000,
            callback=callback,
            enable_intermediate_result=True,
        )
        recognition.start()
        logger.info(f"✅ ASR 会话已创建: {client_id}")
        await websocket.send_text(json.dumps({"type": "asr_ready", "msg": "语音识别已就绪"}))
    except Exception as e:
        logger.error(f"ASR 启动失败: {e}")
        await websocket.send_text(json.dumps({"type": "asr_error", "msg": f"ASR 启动失败: {e}"}))
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "set_language":
                target_lang = message.get("target_lang", "en")
                rooms[room_id]["languages"][client_id] = target_lang
                logger.info(f"   {client_id} 目标语言: {target_lang}")
                await broadcast_room_status(room_id)

            elif msg_type == "audio":
                audio_b64 = message.get("audio", "")
                if not audio_b64:
                    continue
                pcm_bytes = base64.b64decode(audio_b64)

                if callback.is_broken:
                    logger.warning(f"ASR 会话已断开，尝试重建: {client_id}")
                    try:
                        recognition = Recognition(
                            model=ASR_MODEL,
                            format="pcm",
                            sample_rate=16000,
                            callback=callback,
                            enable_intermediate_result=True,
                        )
                        recognition.start()
                        callback.is_broken = False
                        logger.info(f"✅ ASR 会话已重建: {client_id}")
                    except Exception as e:
                        logger.error(f"ASR 重建失败: {e}")
                        continue

                try:
                    recognition.send_audio_frame(pcm_bytes)
                except Exception as e:
                    logger.error(f"发送音频失败: {e}")
                    callback.is_broken = True

    except WebSocketDisconnect:
        logger.info(f"❌ 客户端 {client_id} 断开连接")
    finally:
        try:
            recognition.stop()
        except Exception:
            pass
        if room_id in rooms:
            rooms[room_id]["clients"].pop(client_id, None)
            rooms[room_id]["languages"].pop(client_id, None)
            if not rooms[room_id]["clients"]:
                await asyncio.sleep(5)
                if room_id in rooms and not rooms[room_id]["clients"]:
                    del rooms[room_id]
            else:
                await broadcast_room_status(room_id)


# ── 处理识别结果 ──────────────────────────────────────────
async def handle_asr_result(client_id: str, text: str, room_id: str, is_end: bool):
    if room_id not in rooms:
        return
    if client_id not in rooms[room_id]["clients"]:
        return

    speaker_ws = rooms[room_id]["clients"][client_id]
    try:
        await speaker_ws.send_text(json.dumps({
            "type": "asr_result",
            "text": text,
            "is_end": is_end
        }))
    except Exception as e:
        logger.error(f"发送识别结果失败: {e}")

    if not is_end:
        return

    target_langs = {
        cid: lang for cid, lang in rooms[room_id]["languages"].items()
        if cid != client_id
    }
    if target_langs:
        await asyncio.gather(*[
            translate_and_synthesize(text, target_lang, target_cid, room_id, client_id)
            for target_cid, target_lang in target_langs.items()
        ])


# ── 翻译 + TTS ──────────────────────────────────────────
async def translate_and_synthesize(text: str, target_lang: str,
                                   target_client_id: str, room_id: str,
                                   speaker_id: str):
    try:
        translated = await translate_text(text, target_lang)
        audio_bytes = await synthesize_speech(translated)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else ""

        if room_id in rooms and target_client_id in rooms[room_id]["clients"]:
            target_ws = rooms[room_id]["clients"][target_client_id]
            await target_ws.send_text(json.dumps({
                "type": "translation",
                "from": speaker_id,
                "text": translated,
                "audio": audio_b64,
                "lang": target_lang
            }))
    except Exception as e:
        logger.error(f"翻译合成失败: {e}")


# ── 广播房间状态 ──────────────────────────────────────────
async def broadcast_room_status(room_id: str):
    if room_id not in rooms:
        return
    clients = rooms[room_id]["clients"]
    languages = rooms[room_id]["languages"]
    status = {
        "type": "room_status",
        "clients": [
            {"id": cid, "lang": languages.get(cid, "未设置")}
            for cid in clients.keys()
        ]
    }
    for ws in clients.values():
        try:
            await ws.send_text(json.dumps(status))
        except Exception:
            pass


# ── 启动入口 ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

import os
import json
import time
import uuid
import base64
import asyncio
import logging
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dashscope.audio.http_tts.http_speech_synthesizer import HttpSpeechSynthesizer
import requests
import io
import wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# ⚠️ 架构说明（2026-08-30 改版）
# 原来用的是百炼工作空间专属的实时语音互译端点
# （qwen3-translation-realtime，WORKSPACE_ID.REGION.xxx.aliyuncs.com），
# 实测该端点会拒绝/重置境外（非中国大陆）IP 的连接（code=1006）。
# Render 服务器不在中国大陆，所以这条路走不通。
#
# 改用 dashscope.aliyuncs.com 域名下的三个标准公开接口：
# 流式语音识别（Recognition）+ 机器翻译（REST）+ 语音合成（TTS SDK），
# 这三个接口经实测在境外 IP 下可以正常访问。
# 代价：识别→翻译→合成变成串行三步，延迟比原来的一体化实时模型更高。
# ══════════════════════════════════════════════════════════

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️ 环境变量 DASHSCOPE_API_KEY 未设置！")

ASR_MODEL = "fun-asr-realtime"
TRANSLATE_MODEL = "qwen-mt-turbo"
TTS_MODEL = "cosyvoice-v2"
TTS_VOICE = "longxiaochun_v2"

# qwen-mt 系列的 translation_options.target_lang 要求完整英文语言名，
# 不接受 "en"/"zh" 这种短代码，这里做一层映射（覆盖前端语言下拉框的 12 种语言）。
LANG_NAME_MAP = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian",
    "ar": "Arabic", "th": "Thai", "vi": "Vietnamese", "pt": "Portuguese",
}

# 音频保活看门狗参数（沿用已验证可行的设计）
WATCHDOG_CHECK_INTERVAL = 5
WATCHDOG_IDLE_TIMEOUT = 15

rooms: Dict[str, Dict] = {}   # room_id -> {"a": {...}, "b": {...}}


# ---------- ASR 回调 ----------
class StreamingCallback(RecognitionCallback):
    def __init__(self, room_id: str, role: str, loop: asyncio.AbstractEventLoop):
        self.room_id = room_id
        self.role = role
        self.loop = loop
        self.is_broken = False

    def on_open(self):
        logger.info(f"ASR 流式会话已建立: {self.room_id}/{self.role}")

    def on_close(self):
        self.is_broken = True
        logger.info(f"ASR 流式会话已关闭: {self.room_id}/{self.role}")

    def on_complete(self):
        logger.info(f"ASR 流式会话正常结束: {self.room_id}/{self.role}")

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
        logger.error(f"ASR 错误 [{self.room_id}/{self.role}]: {error_msg}")

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
        asyncio.run_coroutine_threadsafe(
            handle_asr_result(self.room_id, self.role, text, is_end),
            self.loop
        )


# ---------- 翻译（阻塞调用，丢线程池） ----------
def _translate_text_blocking(text: str, target_lang: str) -> str:
    if not text or not DASHSCOPE_API_KEY:
        return text
    target_lang_name = LANG_NAME_MAP.get(target_lang, target_lang)
    try:
        response = dashscope.Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model=TRANSLATE_MODEL,
            messages=[{"role": "user", "content": text}],
            translation_options={"source_lang": "auto", "target_lang": target_lang_name},
            result_format="message",
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        logger.error(f"翻译请求失败: status={response.status_code}, "
                     f"code={getattr(response, 'code', '')}, message={getattr(response, 'message', '')}")
        return text
    except Exception as e:
        logger.error(f"翻译异常: {e}")
        return text


async def translate_text(text: str, target_lang: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _translate_text_blocking, text, target_lang)


# ---------- TTS（阻塞调用，丢线程池） ----------
# ⚠️ 2026-08-31 改版：cosyvoice-v2 原本走的 tts_v2.SpeechSynthesizer 是 WebSocket
# 长连接，实测在 Render（境外 IP）上连接建立失败（self.sock 一直是 None），
# 报 'NoneType' object has no attribute 'close_frame'。
# 改用非流式的 HTTP 版本 HttpSpeechSynthesizer（走普通 REST 请求，跟已验证
# 可用的翻译接口是同一类），返回一个 audio_url，再下载一次拿到音频字节。
def _synthesize_speech_blocking(text: str) -> Optional[bytes]:
    if not text or not DASHSCOPE_API_KEY:
        return None
    last_err = None
    for attempt in range(1, 3):
        try:
            result = HttpSpeechSynthesizer.call(
                model=TTS_MODEL,
                text=text,
                voice=TTS_VOICE,
                format="wav",
                sample_rate=24000,
                stream=False,
                api_key=DASHSCOPE_API_KEY,
            )
            audio_url = getattr(result, "audio_url", None)
            if not audio_url:
                logger.warning(f"TTS 第{attempt}次调用未返回 audio_url，result={result}")
                continue
            resp = requests.get(audio_url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"TTS 第{attempt}次下载音频失败: status={resp.status_code}")
                continue
            # 前端按裸 16bit PCM 解码播放（不认 WAV 头），这里用 wave 模块
            # 正确解析出音频帧数据，剥掉头部，避免开头一小段杂音/爆音。
            try:
                with wave.open(io.BytesIO(resp.content), "rb") as wf:
                    pcm_data = wf.readframes(wf.getnframes())
                return pcm_data
            except Exception as e:
                logger.error(f"TTS 第{attempt}次 WAV 解析失败: {e}")
                continue
        except Exception as e:
            last_err = e
            logger.error(f"TTS 第{attempt}次调用失败: {type(e).__name__}: {e}", exc_info=True)
    logger.error(f"TTS 最终失败（已重试）: {last_err}")
    return None


async def synthesize_speech(text: str) -> Optional[bytes]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _synthesize_speech_blocking, text)


# ---------- 每个连接（A 或 B）的会话状态 ----------
class ConnectionState:
    __slots__ = ("ws", "role", "room_id", "target_lang",
                 "recognition", "callback", "last_audio_time", "current_sid")

    def __init__(self, ws: WebSocket, role: str, room_id: str):
        self.ws = ws
        self.role = role
        self.room_id = room_id
        self.target_lang: str = "en"
        self.recognition: Optional[Recognition] = None
        self.callback: Optional[StreamingCallback] = None
        self.last_audio_time: float = time.monotonic()
        self.current_sid: Optional[str] = None


def _create_recognition(callback: StreamingCallback) -> Recognition:
    recognition = Recognition(
        model=ASR_MODEL,
        format="pcm",
        sample_rate=16000,
        callback=callback,
        enable_intermediate_result=True,
    )
    recognition.start()
    return recognition


def _stop_recognition_blocking(recognition: Recognition):
    try:
        recognition.stop()
    except Exception as e:
        logger.error(f"关闭 ASR 会话失败: {e}")


async def rebuild_recognition(state: ConnectionState, loop: asyncio.AbstractEventLoop, reason: str):
    old = state.recognition
    if old is not None:
        await loop.run_in_executor(None, _stop_recognition_blocking, old)

    new_callback = StreamingCallback(state.room_id, state.role, loop)
    try:
        new_recognition = await loop.run_in_executor(None, _create_recognition, new_callback)
    except Exception as e:
        logger.error(f"ASR 重建失败（{reason}）[{state.room_id}/{state.role}]: {e}")
        state.recognition = None
        state.callback = new_callback
        return

    state.recognition = new_recognition
    state.callback = new_callback
    state.last_audio_time = time.monotonic()
    state.current_sid = None
    logger.info(f"✅ ASR 会话已重建（{reason}）[{state.room_id}/{state.role}]")


async def audio_watchdog(state: ConnectionState, loop: asyncio.AbstractEventLoop):
    try:
        while True:
            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL)
            if state.recognition is None:
                continue
            idle = time.monotonic() - state.last_audio_time
            if idle > WATCHDOG_IDLE_TIMEOUT and not (state.callback and state.callback.is_broken):
                logger.warning(
                    f"⏰ [{state.room_id}/{state.role}] 已 {idle:.1f}s 未收到音频帧，主动重建 ASR 会话"
                )
                await rebuild_recognition(state, loop, reason="看门狗检测到音频中断")
    except asyncio.CancelledError:
        pass


# ---------- FastAPI ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 双麦克风互译服务器启动（标准 ASR+翻译+TTS 架构）")
    yield
    logger.info("🛑 服务器关闭")

app = FastAPI(lifespan=lifespan)
try:
    app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")
except Exception as e:
    logger.warning(f"静态目录挂载失败: {e}")


@app.get("/")
async def get_index():
    return FileResponse("frontend/index.html")


async def send_json(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.error(f"发送消息失败: {e}")


def other_role(role: str) -> str:
    return "b" if role == "a" else "a"


async def maybe_announce_paired(room_id: str):
    room = rooms.get(room_id)
    if not room or "a" not in room or "b" not in room:
        return
    for r in ("a", "b"):
        await send_json(room[r]["ws"], {
            "kind": "status",
            "paired": True,
            "text": "✅ 已连接，点击麦克风开始说话（静音自动断句）"
        })


@app.websocket("/ws/{room_id}/{role}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, role: str):
    if role not in ("a", "b"):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    loop = asyncio.get_running_loop()
    state = ConnectionState(websocket, role, room_id)

    rooms.setdefault(room_id, {})
    rooms[room_id][role] = {"ws": websocket, "state": state}
    logger.info(f"角色 {role} 连入房间 {room_id}")

    await send_json(websocket, {"kind": "status", "paired": False, "text": "⏳ 等待另一方连接..."})

    watchdog_task = None

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "init":
                state.target_lang = message.get("target_lang", "en")
                logger.info(f"[{room_id}/{role}] 目标语言 -> {state.target_lang}")

                if state.recognition is None:
                    try:
                        state.callback = StreamingCallback(room_id, role, loop)
                        state.recognition = await loop.run_in_executor(
                            None, _create_recognition, state.callback
                        )
                        state.last_audio_time = time.monotonic()
                        logger.info(f"✅ ASR 会话已创建 [{room_id}/{role}]")
                    except Exception as e:
                        logger.error(f"ASR 启动失败 [{room_id}/{role}]: {e}")
                        await send_json(websocket, {"kind": "status", "paired": False,
                                                     "text": f"❌ ASR 启动失败: {e}"})
                    watchdog_task = asyncio.create_task(audio_watchdog(state, loop))

                await maybe_announce_paired(room_id)

            elif msg_type == "audio":
                audio_b64 = message.get("data", "")
                if not audio_b64 or state.recognition is None:
                    continue
                pcm_bytes = base64.b64decode(audio_b64)
                state.last_audio_time = time.monotonic()

                if state.callback and state.callback.is_broken:
                    await rebuild_recognition(state, loop, reason="主动检测")
                    if state.recognition is None:
                        continue

                try:
                    await loop.run_in_executor(None, state.recognition.send_audio_frame, pcm_bytes)
                except Exception as e:
                    logger.error(f"发送音频失败 [{room_id}/{role}]: {e}")
                    if state.callback:
                        state.callback.is_broken = True

            elif msg_type == "vad_stop":
                # ASR 自带服务端断句检测（sentence_end），这里仅记录，不强制动作。
                logger.info(f"📢 收到 {role} 的 vad_stop（前端静音检测，ASR 自行断句）")

    except WebSocketDisconnect:
        logger.info(f"角色 {role} 离开房间 {room_id}")
    finally:
        if watchdog_task:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
        if state.recognition:
            await loop.run_in_executor(None, _stop_recognition_blocking, state.recognition)
        room = rooms.get(room_id)
        if room:
            room.pop(role, None)
            if not room:
                rooms.pop(room_id, None)
                logger.info(f"房间 {room_id} 已清理")


# ---------- 处理 ASR 结果：识别 -> 翻译 -> 合成 -> 回传给说话者自己那条连接 ----------
async def handle_asr_result(room_id: str, role: str, text: str, is_end: bool):
    room = rooms.get(room_id)
    if not room or role not in room:
        return
    conn = room[role]
    ws = conn["ws"]
    state: ConnectionState = conn["state"]

    if state.current_sid is None:
        state.current_sid = uuid.uuid4().hex

    sid = state.current_sid

    # 中间结果：只做实时字幕滚动，不翻译不合成
    await send_json(ws, {"kind": "self_original", "text": text, "final": is_end, "sid": sid})

    if not is_end:
        return

    # 整句说完，进入翻译+合成
    try:
        translated = await translate_text(text, state.target_lang)
        await send_json(ws, {"kind": "self_translation", "text": translated, "final": True, "sid": sid})

        audio_bytes = await synthesize_speech(translated)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await send_json(ws, {"kind": "self_audio", "audio": audio_b64, "sid": sid})
    except Exception as e:
        logger.error(f"翻译/合成失败 [{room_id}/{role}]: {e}")
    finally:
        await send_json(ws, {"kind": "audio_done", "sid": sid})
        state.current_sid = None

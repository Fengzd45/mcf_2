"""
双向实时语音互译桥接服务 — AB双麦克风键版
设计原则：
  - A点A键、B点B键，各自声明说话权
  - 扬声器串行播放：原音→译音，播完才解锁对方键
  - 麦克风冲突由人解决（听到沉默才点键）
  - 扬声器冲突由系统解决（串行队列）
  - 两个逻辑通道，同一时刻只有一个工作，只付一份token
"""
import asyncio
import json
import os
import sys
import uuid
import time
import base64
from typing import Dict, Optional
from datetime import datetime

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = sys.stderr if level in ["ERROR", "WARNING"] else sys.stdout
    print(f"[{timestamp}] [{level}] {msg}", file=out)
    sys.stdout.flush()

# ── 环境变量 ──────────────────────────────────────────────
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "")
REGION = os.environ.get("REGION", "cn-beijing")
MODEL = os.environ.get("MODEL", "qwen3-translation-realtime-v1")
SILENCE_PCM = base64.b64encode(b"\x00" * 3200).decode()

if not DASHSCOPE_API_KEY or not WORKSPACE_ID:
    log("⚠️ DASHSCOPE_API_KEY 或 WORKSPACE_ID 未设置!", "WARNING")

# ── FastAPI ──────────────────────────────────────────────
app = FastAPI()

# ── 房间管理 ──────────────────────────────────────────────
class Room:
    def __init__(self, room_id: str):
        self.id = room_id
        self.clients: Dict[str, WebSocket] = {}       # role -> websocket
        self.langs: Dict[str, str] = {}               # role -> lang_code
        self.upstreams: Dict[str, "UpstreamSession"] = {}
        self._last_translation = ""

    def other(self, role: str) -> str:
        return "b" if role == "a" else "a"

    def cleanup(self):
        for up in self.upstreams.values():
            asyncio.create_task(up.finish())
        self.upstreams.clear()

# ── 安全发送 ──────────────────────────────────────────────
async def safe_send(ws: Optional[WebSocket], payload: dict):
    if ws is None:
        return
    try:
        await ws.send_text(json.dumps(payload))
    except Exception as e:
        log(f"safe_send 失败: {e}", "WARNING")

# ── UpstreamSession ───────────────────────────────────────
class UpstreamSession:
    """
    一个方向的端对端翻译通道。
    mic_role 说话 → 译文、译音都发回 mic_role 自己
    """
    def __init__(self, room: Room, mic_role: str):
        self.room      = room
        self.mic_role  = mic_role
        self.peer_role = room.other(mic_role)
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task:      Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connected  = False
        self._is_active  = True
        self._last_audio_time   = 0.0
        self._last_src_text     = ""   # 原文去重
        self._last_tgt_text     = ""   # 译文去重
        self._session_id = f"up_{mic_role}_{uuid.uuid4().hex[:6]}"
        # ✅ 句子计数器（用于配对原文和译文）
        self._sentence_counter = 0
        self._current_sentence_id = None   # 当前"正在等待译文"的句子 sid
        self._pending_translation = None   # 暂存先到的译文（还没有对应原文）

    # ── 连接上游 ──────────────────────────────────────────
    async def start(self) -> bool:
        source_lang = self.room.langs.get(self.mic_role, "zh")
        target_lang = self.room.langs.get(self.peer_role, "en")
        log(f"启动上游 {self._session_id}: {source_lang} → {target_lang}")

        if not DASHSCOPE_API_KEY or not WORKSPACE_ID:
            log("DASHSCOPE_API_KEY / WORKSPACE_ID 未配置!", "ERROR")
            return False

        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        url = (f"wss://{WORKSPACE_ID}.{REGION}.maas.aliyuncs.com"
               f"/api-ws/v1/realtime?model={MODEL}")
        try:
            try:
                self.ws = await websockets.connect(
                    url, additional_headers=headers,
                    max_size=None, ping_interval=20, ping_timeout=60)
            except TypeError:
                self.ws = await websockets.connect(
                    url, extra_headers=headers,
                    max_size=None, ping_interval=20, ping_timeout=60)
            self._connected = True
            log(f"✅ 上游连接成功 {self._session_id}")
        except Exception as e:
            log(f"❌ 上游连接失败 {self._session_id}: {e}", "ERROR")
            return False

        cfg = {
            "event_id": f"evt_{uuid.uuid4().hex}",
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format":  "pcm",
                "output_audio_format": "pcm",
                "input_audio_transcription": {
                    "model":    "qwen3-asr-flash-realtime",
                    "language": source_lang,
                },
                "translation": {"language": target_lang},
                "audio": {
                    "input_sample_rate":  16000,
                    "output_sample_rate": 24000,
                },
            },
        }
        try:
            await self.ws.send(json.dumps(cfg))
        except Exception as e:
            log(f"❌ 会话配置发送失败 {self._session_id}: {e}", "ERROR")
            return False

        self._recv_task      = asyncio.create_task(self._recv_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return True

    # ── 心跳静音，防上游超时 ──────────────────────────────
    async def _heartbeat_loop(self):
        while self._connected and self._is_active:
            await asyncio.sleep(2)
            if not self._connected:
                break
            if time.time() - self._last_audio_time > 8:
                try:
                    await self.ws.send(json.dumps({
                        "event_id": f"evt_{uuid.uuid4().hex}",
                        "type":     "input_audio_buffer.append",
                        "audio":    SILENCE_PCM,
                    }))
                except Exception:
                    pass

    # ── 发送音频帧 ────────────────────────────────────────
    async def send_audio(self, pcm_b64: str):
        if not self.ws or not self._connected or not self._is_active:
            return
        self._last_audio_time = time.time()
        try:
            await self.ws.send(json.dumps({
                "event_id": f"evt_{uuid.uuid4().hex}",
                "type":     "input_audio_buffer.append",
                "audio":    pcm_b64,
            }))
        except Exception:
            self._connected = False

    # ── 关闭上游 ──────────────────────────────────────────
    async def finish(self):
        if not self.ws or not self._connected:
            return
        self._is_active = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        try:
            await self.ws.send(json.dumps({
                "event_id": f"evt_{uuid.uuid4().hex}",
                "type":     "session.finish",
            }))
            await asyncio.wait_for(self.ws.wait_closed(), timeout=5)
        except (asyncio.TimeoutError, Exception):
            pass
        finally:
            self._connected = False
            if self._recv_task:
                self._recv_task.cancel()

    # ── 文本有效性检查 ────────────────────────────────────
    def _is_valid(self, text: str) -> bool:
        if not text or len(text.strip()) < 2:
            return False
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
        has_lat = any('a' <= c.lower() <= 'z' for c in text)
        return has_cjk or has_lat

    # ── 接收上游事件循环 ──────────────────────────────────
    async def _recv_loop(self):
        try:
            async for raw in self.ws:
                if not self._is_active:
                    break
                try:
                    event = json.loads(raw)
                    etype = event.get("type", "")

                    mic_ws  = self.room.clients.get(self.mic_role)   # 说话者
                    peer_ws = self.room.clients.get(self.peer_role)  # 对方

                    # ── 原文流式（实时字幕给说话者看）────────
                    if etype == "conversation.item.input_audio_transcription.text":
                        text = event.get("text", "")
                        await safe_send(mic_ws, {
                            "kind": "self_original",
                            "text": text,
                            "final": False
                        })

                    # ── 原文最终 ─────────────────────────────
                    elif etype == "conversation.item.input_audio_transcription.completed":
                        text = event.get("transcript", "").strip()
                        if not self._is_valid(text):
                            log(f"⚠️ 过滤无效原文: {repr(text)}")
                            continue
                        if text == self._last_src_text:
                            continue
                        self._last_src_text = text

                        # ✅ 生成句子ID
                        self._sentence_counter += 1
                        self._current_sentence_id = self._sentence_counter

                        log(f"✅ [{self._session_id}] 原文 #{self._current_sentence_id}: {text}")
                        await safe_send(mic_ws, {
                            "kind": "self_original",
                            "text": text,
                            "final": True,
                            "sid": self._current_sentence_id
                        })

                        # ✅ 如果有暂存的译文，立即发送并配对
                        if self._pending_translation:
                            log(f"📤 发送暂存译文 #{self._current_sentence_id}")
                            await safe_send(mic_ws, {
                                "kind": "self_translation",
                                "text": self._pending_translation,
                                "final": True,
                                "sid": self._current_sentence_id
                            })
                            self._pending_translation = None
                            # ✅ 这句已经配完对了，立刻释放 sid 槽位
                            self._current_sentence_id = None

                    # ── 译文流式（发给说话者自己）────────────
                    elif etype in ("response.audio_transcript.text", "response.text.text"):
                        text = event.get("text", "")
                        await safe_send(mic_ws, {
                            "kind": "self_translation",
                            "text": text,
                            "final": False
                        })

                    # ── 译文最终（发给说话者自己）────────────
                    elif etype in ("response.audio_transcript.done", "response.text.done"):
                        text = event.get("transcript") or event.get("text") or ""
                        text = text.strip()
                        if not self._is_valid(text):
                            log(f"⚠️ 过滤无效译文: {repr(text)}")
                            continue
                        if text == self._last_tgt_text:
                            continue
                        if text == self.room._last_translation:
                            log(f"⚠️ 全局重复译文: {repr(text)}")
                            continue
                        self._last_tgt_text = text
                        self.room._last_translation = text

                        # ✅ 如果还没有原文（sid 槽位空着），暂存译文
                        if self._current_sentence_id is None:
                            self._pending_translation = text
                            log(f"📝 暂存译文（等待原文）: {text}")
                            continue

                        log(f"✅ [{self._session_id}] 译文 #{self._current_sentence_id}: {text}")
                        await safe_send(mic_ws, {
                            "kind": "self_translation",
                            "text": text,
                            "final": True,
                            "sid": self._current_sentence_id
                        })
                        # ✅✅ 关键修复：用完立即清空 sid 槽位。
                        # 若不清空，下一句的译文如果比它自己的原文
                        # 更早到达，会被误判为"槽位已占用"从而错误地
                        # 绑定到上一句已经配对完成的 sid 上，
                        # 导致前端出现"原文栏显示英文/卡片配对错乱"。
                        self._current_sentence_id = None

                    # ── 译音PCM（发给说话者自己播放）─────────
                    elif etype == "response.audio.delta":
                        audio = event.get("delta", "")
                        if audio:
                            await safe_send(mic_ws, {
                                "kind": "self_audio",
                                "audio": audio
                            })

                    # ── 译音播放结束通知 ──────────────────────
                    elif etype == "response.audio.done":
                        await safe_send(mic_ws, {"kind": "audio_done"})
                        log(f"🔔 [{self._session_id}] 译音完成，通知 {self.mic_role} 解锁")

                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._connected = False

# ── 房间存储 ──────────────────────────────────────────────
rooms: Dict[str, Room] = {}

# ── WebSocket 端点 ────────────────────────────────────────
@app.websocket("/ws/{room_id}/{role}")
async def ws_endpoint(websocket: WebSocket, room_id: str, role: str):
    if role not in ("a", "b"):
        await websocket.close()
        return

    await websocket.accept()
    room = rooms.setdefault(room_id, Room(room_id))
    room.clients[role] = websocket

    try:
        # ── 握手：等待 init 消息 ──────────────────────────
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        init = json.loads(raw)
        if init.get("type") != "init":
            await websocket.close(code=4002)
            return

        room.langs[role] = init.get("lang", "zh")
        peer_lang = init.get("target_lang", "en")
        if room.other(role) not in room.langs:
            room.langs[room.other(role)] = peer_lang

        log(f"角色 {role} 连入房间 {room_id}, 语言={room.langs[role]}")

        # ── 启动本角色的上游通道 ─────────────────────────
        key = f"{role}2{room.other(role)}"
        up  = UpstreamSession(room, mic_role=role)
        ok  = await up.start()
        if not ok:
            await websocket.close(code=4005)
            return
        room.upstreams[key] = up

        await safe_send(websocket, {
            "kind": "status", "text": "✅ 已连接", "paired": True
        })

        # ── 主循环：接收前端音频帧和控制消息 ──────────────
        while True:
            try:
                msg_raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=120.0)
                msg = json.loads(msg_raw)

                if msg.get("type") == "audio":
                    up_session = room.upstreams.get(key)
                    if up_session:
                        await up_session.send_audio(msg["data"])

                # ── VAD 静音断句 ──────────────────────────
                elif msg.get("type") == "vad_stop":
                    up_session = room.upstreams.get(key)
                    if up_session and up_session.ws and up_session._connected:
                        try:
                            await up_session.ws.send(json.dumps({
                                "event_id": f"evt_{uuid.uuid4().hex}",
                                "type": "input_audio_buffer.commit",
                            }))
                            log(f"📢 VAD 静音断句: {role}")
                        except Exception as e:
                            log(f"VAD断句失败: {e}", "WARNING")

            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                log(f"主循环异常 {role}: {e}", "WARNING")
                break

    finally:
        room.clients.pop(role, None)
        log(f"角色 {role} 离开房间 {room_id}")
        if not room.clients:
            room.cleanup()
            rooms.pop(room_id, None)
            log(f"房间 {room_id} 已清理")
        try:
            await websocket.close()
        except Exception:
            pass

# ── 静态前端 ──────────────────────────────────────────────
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception as e:
    log(f"⚠️ 前端静态文件挂载失败: {e}", "WARNING")

# ── 启动入口（支持环境变量 PORT）─────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# 双向实时语音互译 — AB双麦克风键版

## 设计原则

| 冲突 | 解决方 | 机制 |
|------|--------|------|
| 麦克风冲突（两人抢录） | 人 | 听到扬声器沉默才点己方键 |
| 扬声器冲突（原音/译音顺序） | 系统 | 串行播放队列，原音→译音 |
| 回声（扬声器声进麦克风） | 系统 | 浏览器 AEC（echoCancellation） |
| 键误触（播放中乱点） | 系统 | 播放期间两键全部锁定 |

## 使用流程

```
选语言 → 开始通话 → A点A键说话 → 停顿后VAD截止
→ 播原音 → 播译音 → 沉默
→ B听到沉默点B键说话 → ...
```

## 环境变量

```bash
export DASHSCOPE_API_KEY=sk-xxx
export DASHSCOPE_WORKSPACE_ID=ws-xxx
export BAILIAN_REGION=cn-beijing          # 可选，默认 cn-beijing
export LIVETRANSLATE_MODEL=qwen3.5-livetranslate-flash-realtime  # 可选
```

## 安装 & 启动

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000

## 关键修复（相对原始单键版）

### 后端 main.py
1. **音频路由修复**：原版把A的音频同时送入a2b和b2a两个通道，改为只送属于自己的通道
2. **语言配置修复**：各角色只写自己的语言，不覆盖对方
3. **transcript变量分离**：原文/译文各用独立去重变量，不互相干扰
4. **心跳静音时机**：由3秒改为8秒，避免说话停顿时VAD被静音包干扰
5. **audio_done事件**：新增`response.audio.done`处理，通知前端译音真正播完

### 前端 index.html
1. **双键设计**：A键/B键分开，点击即声明说话权，无需猜角色
2. **锁定机制**：播放中两键全锁，录音中对方键锁定
3. **角色不再靠语种猜**：哪个键触发 → 哪个ws发音频 → 哪个通道处理
4. **audio_done驱动解锁**：由服务端事件而非前端定时器决定何时解锁
5. **看门狗兜底**：8秒内未收到audio_done → 强制解锁（防止事件丢失卡死）
6. **switchTime逻辑删除**：原版切换角色时600ms内双发，新版不需要

## 文件结构

```
project/
├── main.py          # FastAPI 后端
├── requirements.txt
├── README.md
└── frontend/
    └── index.html   # 前端（单文件）
```
# Code Walkthrough — Presentation Flow

Two pathways. The **WebSocket pathway** runs continuously in the background, building up context. The **chat pathway** fires once per user message and reads that context.

---

## The Core Idea

The system is always watching and listening. By the time the user sends a chat message, the model service already knows what emotion has been on their face for the last N seconds, and has a timestamped record of everything they said out loud. The chat handler reads both and hands them to the LLM.

---

## Pathway 1 — WebSocket (background, continuous)

**Entry point:** browser opens `ws://localhost:8000/ws` on page load.

```
frontend/+page.svelte
  ↓  sends {type: "session_start", profile_id}

ws/handler.py  handle_websocket()
  → dispatch table routes to on_session_start()
  → creates HarnessSession in _sessions[profile_id]
     HarnessSession holds:
       - emotion_buffer   (rolling deque, last 10 observations)
       - transcript_buffer (list, last 20 segments)
```

From here, two streams run independently:

### Video stream (every 500 ms)

```
frontend/+page.svelte
  captureVideoFrame()
  → canvas.drawImage(video) → toDataURL("image/jpeg") → base64
  ↓  sends {type: "video_frame", data: base64JPEG, timestamp}

ws/handler.py  on_video_frame()       ← COMPOSITION POINT — read this to follow the pipeline
  │
  ├─ ws/video.py  detect_from_message(face_detector, msg, frame_count)
  │    ├─ decode_frame(data)           base64 → BGR numpy array
  │    └─ run_face_detection(...)      YOLO → face_crop, bounding box, annotated frame
  │         core/face_detector.py  FaceDetector.detect_best(frame_bgr)
  │           ultralytics YOLOv8 → highest-confidence face crop (BGR uint8)
  │         returns FrameDetectionResult
  │
  ├─ pick_emotion(face_crop, emotion_model, detected)    ← only place model is called
  │    if real model loaded + TEST_EMOTIONS=false:
  │      emotion_model.predict(face_crop)  → (label, confidence)
  │    else if TEST_EMOTIONS=true (default):
  │      random.choice(EMOTIONS)           ← placeholder until real model integrated
  │    else:
  │      "neutral", 0.5
  │
  ├─ session.emotion_buffer.update(emotion, confidence, timestamp)
  │    core/emotion/buffer.py  — rolling deque(maxlen=10)
  │    this is what the chat handler reads later
  │
  └─ _send_frame_messages(websocket, session, result, emotion, confidence, timestamp)
       sends: face_detection, frame_debug, emotion_update  → browser updates UI
```

### Audio stream (when speech detected)

```
frontend/browserVad.ts
  VAD polls mic RMS every 100 ms
  → speech detected → MediaRecorder captures WebM/Opus
  → silence for 600 ms → stop recording
  → if clip > 1200 ms: send {type: "audio_chunk", data: base64WebM, timestamp}

ws/handler.py  on_audio_chunk()
  → asyncio.create_task(process_audio_chunk(...))   ← non-blocking

ws/audio.py  process_audio_chunk()
  ├─ decode_browser_audio_to_numpy(data)
  │    ffmpeg: WebM/Opus → float32 PCM (mono, 16 kHz)
  ├─ stt.transcribe(audio_np)
  │    core/stt/whisper_cpp.py  WhisperCppTranscriptionService
  │    → (text, language, confidence)
  ├─ session.transcript_buffer.append(TranscriptSegment(text, timestamp))
  │    this is what the chat handler reads later
  └─ sends: transcript_chunk → browser displays text
```

---

## Pathway 2 — Chat (user-triggered, per message)

**Entry point:** user types a message and hits send.

```
frontend/ChatInput.svelte
  api.sendChat(text)
  ↓  fetch POST /api/v1/chat  (to SvelteKit server-side)
  ↓  SvelteKit proxies to Express

backend/src/routes/chat.router.ts
  → reads profileId from session cookie
  → profileStore.getHistory(profileId)    reads data/profiles/<uuid>.json
  → slices last 20 messages as history
  ↓  fetch POST http://localhost:8000/api/v1/chat

model_service/routers/chat.py  chat()
  │
  ├─ get_session(profile_id)              finds the live WS session in _sessions
  │    → HarnessSession with filled emotion_buffer + transcript_buffer
  │
  ├─ emotion_observations = session.emotion_buffer.history()
  │    → list[EmotionObservation(emotion, confidence, timestamp)]
  │
  ├─ transcript_segments = session.transcript_buffer[-20:]
  │    → list[TranscriptSegment(text, timestamp)]
  │
  ├─ emotion_agent.analyse(emotion_observations, transcript_segments)
  │    core/emotional_reasoning_agent.py
  │    → statistics.mode over labels → dominant emotion
  │    → returns context string e.g.
  │      "The user appears to be feeling happy (~12s). Calibrate tone accordingly."
  │    [TODO: weight by confidence, use transcript cues, richer output]
  │
  ├─ llm_agent.reason(message, emotional_context, history, transcript_segments)
  │    core/llm/reasoning_agent.py
  │    assembles message list:
  │      [system: SYSTEM_PROMPT]           ← empathy bot persona (TBD)
  │      [user/assistant: ...history...]   ← last 10 turns
  │      [system: emotional_context]       ← from EmotionalReasoningAgent
  │      [system: transcript context]      ← timestamped STT segments
  │      [user: current message]
  │    → llm.chat(messages)
  │         core/llm/openai.py  OpenAIProvider
  │         → OpenAI Chat Completions API → reply string
  │
  └─ returns ChatResponse(response=reply)

backend/src/routes/chat.router.ts
  → profileStore.appendMessage(profileId, userMsg)
  → profileStore.appendMessage(profileId, agentMsg)
  → res.json({response})

frontend/ChatHistory.svelte  renders the new turn
```

---

## Where Things Are Not Yet Done

| What | Where | Status |
|---|---|---|
| Real emotion model | `core/emotion/` — implement ABC, register in factory, set env vars | Placeholder (random output) |
| Emotion reasoning | `core/emotional_reasoning_agent.py` | Mode-only, ignores confidence + transcript |
| LLM prompt engineering | `core/llm/reasoning_agent.py` — 5 open TODOs in module docstring | Static, needs decisions |
| Transcript → reasoning | `transcript_segments` is passed but barely used | Low-hanging fruit |

---

## Key Files to Open During the Walkthrough

| File | Why |
|---|---|
| `ws/handler.py` `on_video_frame()` | The clearest example of the composition pattern — read the 5 steps |
| `ws/video.py` | Atomic frame utilities — no emotion, no WS state |
| `ws/handler.py` `pick_emotion()` | The only place the emotion model is invoked |
| `core/emotion/buffer.py` | What accumulates between chat messages |
| `routers/chat.py` | Where both buffers are read and fed to the LLM |
| `core/llm/reasoning_agent.py` | Where the prompt is assembled — all open TODOs are here |

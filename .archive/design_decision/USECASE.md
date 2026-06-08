# EmpathyBot — Project Context

> Upload this file at the start of any Claude conversation to provide full project context.
> Then ask for code, a presentation, a report, UI designs, or any other deliverable.

---

## Product name
**EmpathyBot** — AI-powered patient feedback for GP practices

## One-line pitch
A post-appointment feedback tool that detects emotional mismatches between what patients say and how they feel, then uses an LLM to reassure, explain, and invite genuine concern — especially around AI use in healthcare.

---

## The problem
GPs are increasingly using AI tools (symptom checkers, note summarisers, diagnostic aids). Older and non-technical patients often leave appointments confused, anxious, or mistrustful — but don't know how to voice it. Standard feedback forms miss this entirely.

---

## The solution — 3-layer flow

### Layer 1 — Structured questionnaire
5–7 simple questions about the visit experience. Optimised for low digital literacy: large buttons, plain English, audio option.

### Layer 2 — Emotion-aware response detection
Webcam or interaction-pattern signals (hesitation time, re-selections, micro-expressions via on-device model) are compared against the chosen answer. If a patient selects "I feel fine" but signals discomfort, the system flags a mismatch and triggers an LLM explanation prompt.

### Layer 3 — Pre-submit LLM chat
Before submission, an open input (text or voice) lets the patient ask any question. The LLM responds based on emotion state — reassuring, warm, non-technical — and explains the AI's role in their appointment in plain terms.

---

## Entry modes
- **Waiting room kiosk** — single shared device, session resets after each use
- **GP email link** — personalised URL sent post-appointment, opens on patient's phone

---

## Key personas

| Persona | Age | Tech comfort | Main concern |
|---|---|---|---|
| Margaret | 74 | Very low | "Is the computer making decisions about me?" |
| Raj | 68 | Low-medium | "Who sees my answers?" |
| Carol | 81 | None | "I don't want to press the wrong thing" |

---

## UI design decisions
- Large tap targets, minimal choices per screen
- Plain English everywhere — no "AI", "algorithm", or "data processing" language in the UI itself
- Emotion detection surfaces quietly (amber nudge), never accusatory
- Anonymous by default, stated clearly at the end
- Three screens: Entry → Questionnaire → Pre-submit question box

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React (single page, large-touch UI) |
| Emotion detection | Face-api.js (on-device, no cloud) or interaction heuristics (dwell time, re-clicks) |
| LLM | Anthropic Claude API (`claude-sonnet-4-20250514`) |
| Backend | Node.js + Express (session handling, feedback storage) |
| Database | PostgreSQL (anonymised responses) or JSON store for MVP |
| Auth/link | GP subscribes via dashboard → generates unique feedback URL per patient session |
| Audio input | Web Speech API (browser-native, no extra dependency) |

---

## LLM system prompt (core)

```
You are a friendly, patient assistant helping an older adult understand what happened during their GP visit.
The patient may be confused or anxious about AI being used by their doctor.
Always respond in plain English, no medical or technical jargon.
Keep responses under 3 sentences. Be warm, calm, and reassuring.
Never minimise their concern — validate it, then explain simply.
If the patient's emotion score shows discomfort, open with empathy before any explanation.
```

---

## Emotion-mismatch logic (pseudocode)

```js
if (selectedAnswer === positiveOption && emotionScore < threshold) {
  triggerMismatchAlert()
  llmPrompt = buildPrompt(questionContext, emotionState: "anxious", patientAnswer)
  showLLMExplanation(llmPrompt)
  showFollowUpInvite()
}
```

---

## Data model (simplified)

```json
{
  "sessionId": "uuid",
  "gpPracticeId": "gp_123",
  "timestamp": "ISO8601",
  "responses": [
    { "questionId": "q1", "answer": "mostly", "emotionScore": 0.72, "mismatch": false },
    { "questionId": "q3", "answer": "i_feel_fine", "emotionScore": 0.31, "mismatch": true }
  ],
  "freeTextQuestion": "Will my doctor see this?",
  "llmResponse": "...",
  "submitted": true
}
```

---

## Presentation outline (9 slides)

1. **The gap** — patients leave AI-assisted appointments without understanding what happened
2. **Who we're designing for** — persona cards (Margaret, Raj, Carol)
3. **How it works** — 3-layer flow diagram
4. **Emotion detection** — what it measures, how it stays on-device/private
5. **The LLM moment** — demo of mismatch → explanation → reassurance
6. **GP dashboard** — subscribe, share link, view anonymised trends
7. **Privacy by design** — on-device emotion processing, anonymous by default
8. **Business model** — GP practice subscription (£X/month), scales with list size
9. **What's next** — multilingual support, integration with GP clinical systems

---

## Report outline (10 sections)

1. Executive summary
2. Problem statement — patient trust gap in AI-assisted healthcare
3. User research — older adult digital literacy, post-appointment anxiety
4. Solution design — UX principles, accessibility standards (WCAG 2.1 AA)
5. Technical architecture — emotion detection, LLM integration, data flow
6. Ethical considerations — consent, on-device processing, GDPR compliance
7. Pilot design — how to run a GP practice trial
8. Success metrics — mismatch detection rate, question submission rate, sentiment shift pre/post LLM response
9. Limitations and risks
10. Recommendations

---

## Success metrics
- Mismatch detection rate (emotion vs. selected answer)
- Follow-up question submission rate
- Sentiment shift pre/post LLM response
- GP practice subscription retention
- Patient-reported comfort with AI post-session

---

## Business model
- GP practice subscribes monthly (tiered by patient list size)
- GP generates shareable feedback links from a simple dashboard
- Waiting room kiosk option included in subscription
- Anonymised aggregate insights returned to the GP practice

---

## Ethical and compliance notes
- Emotion detection runs fully on-device — no video or biometric data leaves the browser
- All feedback is anonymous by default
- GDPR compliant — no PII stored without explicit opt-in
- Patients can skip any question or exit at any time
- Consent screen shown before questionnaire begins

---

## What to ask Claude with this file

- "Build the React frontend for the questionnaire screen"
- "Create a 9-slide PowerPoint presentation"
- "Write the full technical report"
- "Design the GP dashboard UI"
- "Write the Node.js backend with session handling"
- "Draft the GDPR consent screen copy"
- "Build the emotion-mismatch detection logic"
- "Create the LLM API integration with the system prompt"

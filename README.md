# CS731 — Emotion-Aware Chatbot

**Group**: [Your group name]  
**Members**: [Member 1], [Member 2], [Member 3]  
**Course**: COMPSYS 731 / CS731, University of Auckland

---

## Project Overview

An emotion-aware chatbot that detects facial expressions in real time via webcam and tailors its conversational responses accordingly. The system uses a custom-trained CNN for emotion recognition and OpenAI's API for the chatbot.

---

## Repository Structure

```
project-root/
├── README.md
├── requirements.txt
├── .env.example              ← copy to .env and add your API key
├── main.py                   ← entry point (terminal application)
├── face_detector.py          ← face detection (YOLOv8, RetinaFace, MediaPipe, Haar)
├── emotion_inferencer.py     ← inference wrapper + emotion buffer
├── chatbot.py                ← LLM integration + multi-model comparison
├── speech.py                 ← FasterWhisper speech-to-text
├── data/
│   ├── dataset_preparation.py  ← build train/val splits from labels.csv
│   └── emotion_dataset.py      ← PyTorch Dataset class + MixUp/CutMix
├── models/
│   ├── model.py                ← all model architectures (timm + ChatBox_V1)
│   ├── train.py                ← training loop with curves + checkpointing
│   ├── evaluate.py             ← confusion matrix, per-class metrics
│   └── checkpoints/            ← saved .pt files (download separately)
├── notebooks/
│   └── CS731_Face_Detection_Comparison.ipynb
├── results/
│   ├── face_detection/
│   ├── training/
│   └── evaluation/
└── weights/
    └── yolov8n-face.pt         ← auto-downloaded on first run
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

### 2. Create and activate environment

```bash
# Using conda (recommended)
conda create -n cs731 python=3.10
conda activate cs731

# Or using venv
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-...
```

### 5. Download trained models

Pre-trained checkpoints are hosted on REANNZ FileSender (too large for GitHub):

```
[Link to REANNZ FileSender — add before submission]
```

Place downloaded `.pt` files in `models/checkpoints/`.

---

## Dataset Setup

Download **AffectNet-HQ** (provided by course) and place it as:

```
1_Dataset/
├── labels.csv
├── anger/
├── contempt/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Then run the preparation script:

```bash
# Ekman's 6 emotions (no contempt, no neutral) — 85/15 split
python data/dataset_preparation.py --mode ekman6

# Ekman's 7 (includes neutral) — 70/15/15 split
python data/dataset_preparation.py --mode ekman7 --test_size 0.15
```

This creates `data/splits/ekman6_train.csv` and `data/splits/ekman6_val.csv`.

---

## Training

```bash
# Train a single model
python models/train.py --model swin_tiny --mode ekman6 --epochs 20

# Train all 5 comparison models (Group 15 style)
python models/train.py --all --mode ekman6 --epochs 20

# ChatBox_V1 (7-class, Team 7 style)
python models/train.py --model chatbox_v1 --mode ekman7 --epochs 30

# With custom learning rate
python models/train.py --model convnext_tiny --lr 0.0001 --epochs 23
```

Checkpoints saved to `models/checkpoints/`.  
Training curves saved to `results/training/`.

---

## Evaluation

```bash
python models/evaluate.py \
  --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt \
  --split val
```

Outputs: accuracy, per-class F1, confusion matrix PNG.

---

## Running the Chatbot

```bash
# Full pipeline (webcam + voice + chatbot)
python main.py --checkpoint models/checkpoints/swin_tiny_ekman6_best.pt --voice

# No webcam (for testing — fixed mock emotion)
python main.py --no_webcam --mock_emotion happy

# No webcam, no voice (keyboard only)
python main.py --no_webcam --no_voice
```

**Controls in terminal:**
- Type your message and press Enter to send
- Press Enter with empty input to record voice (if `--voice` enabled)
- Type `reset` to clear conversation history
- Type `quit` to exit

---

## Face Detection Comparison

Open the notebook to reproduce the face detection comparison:

```bash
jupyter notebook notebooks/CS731_Face_Detection_Comparison.ipynb
```

Results saved to `results/face_detection/`.

---

## LLM Comparison

```python
from chatbot import compare_llms, save_comparison_results

test_msgs = [
    {'text': "I've been feeling quite lonely lately.", 'emotion': 'sad'},
    {'text': "I had a wonderful day today!", 'emotion': 'happy'},
    {'text': "I'm not sure how I feel about everything.", 'emotion': 'neutral'},
]

results = compare_llms(test_msgs, models=['o4-mini', 'gpt-4o', 'o3'])
save_comparison_results(results, 'results/llm_comparison.csv')
```

---

## Environment Variables

```bash
# .env.example — copy to .env and fill in
OPENAI_API_KEY=sk-your-key-here
```

---

## Known Issues & Notes

- **RetinaFace**: has TensorFlow dependency conflicts on Python 3.12+. It is evaluated in the notebook but skipped gracefully in the live pipeline.
- **MediaPipe**: not available for Python 3.13 at time of writing. Skipped gracefully.
- **YOLOv8n-face weights**: auto-downloaded from GitHub on first run. Requires internet connection.
- **Windows Long Paths**: if you hit path-too-long errors on Windows, enable long path support in Group Policy or registry.

---

## References

- AffectNet-HQ: Mollahosseini et al. (2019), IEEE Transactions on Affective Computing
- YOLOv8-face: https://github.com/akanametov/yolov8-face
- timm: Wightman (2019), PyTorch Image Models
- FasterWhisper: https://github.com/SYSTRAN/faster-whisper
- Ekman (1992): An argument for basic emotions

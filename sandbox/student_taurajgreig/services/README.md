# Transcription Services

Pluggable transcription service implementations with isolated dependencies.

## Architecture

Each transcription service:
1. Inherits from `TranscriptionService` base class
2. Implements `transcribe(audio_data) -> (transcript, language)`
3. Declares dependencies via `REQUIRES_DEPS` and `REQUIREMENTS_FILE`

## Available Services

### MockTranscriptionService
- **Status**: ✅ Enabled (no dependencies)
- **Purpose**: Testing VAD logic without loading models
- **Use**: `--service mock`

### WhisperTranscriptionService
- **Status**: ❌ Disabled (memory constrained on this machine)
- **Dependencies**: `faster-whisper`, `torch`, `huggingface-hub`
- **Install**: `make sandbox-init-whisper`
- **Models**: tiny, base, small, medium, large-v3
- **Use**: `--service whisper --model tiny`

### WhisperDistilledTranscriptionService
- **Status**: ❌ Disabled (memory constrained on this machine)
- **Dependencies**: Same as Whisper
- **Install**: `make sandbox-init-whisper`
- **Models**: distil-small.en, distil-medium.en, distil-large-v3
- **Use**: `--service whisper-distilled --model distil-small.en`

## Adding a New Service

### 1. Create service file: `services/my_service.py`

```python
from .base import TranscriptionService
import numpy as np

class MyTranscriptionService(TranscriptionService):
    """My transcription service."""

    REQUIRES_DEPS = ["package1", "package2"]
    REQUIREMENTS_FILE = "requirements-my-service.txt"

    def __init__(self, model_name="default"):
        # Initialize service
        pass

    def transcribe(self, audio_data: np.ndarray) -> tuple[str, str]:
        """Transcribe audio.

        Returns:
            (transcript, language)
        """
        # Implement transcription
        return transcript, language
```

### 2. Create requirements file: `services/requirements-my-service.txt`

```
# My service dependencies
package1>=1.0.0
package2>=2.0.0
```

### 3. Export in `services/__init__.py`

```python
from .my_service import MyTranscriptionService

__all__ = [
    # ... existing ...
    "MyTranscriptionService",
]
```

### 4. Update Makefile: `Makefile`

```makefile
sandbox-init-my-service: sandbox-init
	$(SANDBOX_PYTHON) -m pip install -r sandbox/student_taurajgreig/services/requirements-my-service.txt
	@echo "✓ My service dependencies installed"
```

Add to `.PHONY` and help comments.

### 5. Update main script

```python
# In speech_recognition_test.py
MODEL_SERVICE = "my-service"
MODEL_NAME = "default"
```

Or use CLI:
```bash
python speech_recognition_test.py --service my-service
```

## Dependency Installation

**Core only** (always installed):
```bash
make sandbox-init
```

**With Whisper support**:
```bash
make sandbox-init-whisper
```

**With future services**:
```bash
make sandbox-init-{service-name}
```

## Benefits

✅ No bloated requirements.txt - install only what you need
✅ Services declare their dependencies explicitly
✅ Easy to add new services without updating core files
✅ Clear separation of concerns
✅ Easy to swap services in production

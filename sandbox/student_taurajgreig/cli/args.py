"""Command-line argument parsing and service creation."""

import argparse
from services import (
    MockTranscriptionService,
    WhisperTranscriptionService,
    WhisperDistilledTranscriptionService,
    OpenAIWhisperTranscriptionService,
)


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Speech recognition with voice activation detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_recognition_test.py              # mock (default)
  python speech_recognition_test.py --service whisper-distilled --model distil-small.en --duration 5  # Fixed 5-second recording
  python speech_recognition_test.py --service whisper --model tiny  # Whisper with VAD
        """,
    )
    parser.add_argument(
        "--service",
        type=str,
        default="mock",
        choices=["mock", "whisper", "whisper-distilled", "openai"],
        help="Transcription service (default: mock - no model). Use 'openai' to test with OpenAI Whisper API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=[
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "distil-small.en",
            "distil-medium.en",
            "distil-large-v3",
        ],
        help="Model size for Whisper (default: tiny). Distilled models are optimized for speed.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Record for fixed duration in seconds (e.g., --duration 5). Overrides VAD and debug modes.",
    )
    parser.add_argument(
        "--save-audio",
        type=str,
        default=None,
        help="Save recorded audio to directory (e.g., --save-audio ./audio_debug). Useful for debugging.",
    )

    return parser.parse_args()


def create_service(service_name, model_name=None):
    """Create transcription service instance.

    Args:
        service_name: "mock" | "whisper" | "whisper-distilled" | "openai"
        model_name: Model size (only used for whisper services)

    Returns:
        TranscriptionService: Service instance

    Raises:
        ValueError: If service_name is unknown
    """
    if service_name == "mock":
        print("[INIT] Using MockTranscriptionService (no model loaded)\n")
        return MockTranscriptionService()
    elif service_name == "whisper":
        print(f"[INIT] Using WhisperTranscriptionService ({model_name} model)\n")
        return WhisperTranscriptionService(model_name)
    elif service_name == "whisper-distilled":
        print(
            f"[INIT] Using WhisperDistilledTranscriptionService ({model_name} model)\n"
        )
        return WhisperDistilledTranscriptionService(model_name)
    elif service_name == "openai":
        print("[INIT] Using OpenAIWhisperTranscriptionService (cloud API)\n")
        return OpenAIWhisperTranscriptionService()
    else:
        raise ValueError(f"Unknown service: {service_name}")

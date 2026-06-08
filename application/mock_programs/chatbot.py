"""
CS731, Chatbot Module
========================
Implements:
  1. OpenAI chatbot with emotion-aware system prompt (Group 15 style)
  2. Multi-LLM comparison framework (required: compare ≥ 3 models)
  3. Prompt engineering iterations

Usage
-----
  from chatbot import EmotionChatbot, compare_llms

  bot = EmotionChatbot(model='o4-mini')
  response = bot.chat("I'm feeling a bit down today.", emotion='sad')

Environment
-----------
  Set OPENAI_API_KEY in a .env file or environment variable.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print('[WARN] openai not installed. Run: pip install openai')


# ── System prompts (from Group 15 report, two iterations) ───────────────────

SYSTEM_PROMPT_V1 = """You are Frienderly, a chatbot designed to be a warm, understanding companion \
for elderly users. You speak as if you are their long-time friend: casual, kind, and thoughtful \
without being overly sentimental. Each user message may include a note like '[user is feeling happy]' \
or '[user is feeling sad]' at the end. Do not reference the emotion directly, but use it to guide \
your tone and content. Your responses should be emotionally aware and supportive, naturally flowing \
from what the user says. Maintain a friendly, conversational style that feels authentic and familiar."""

SYSTEM_PROMPT_V2 = """You are Frienderly, a chatbot designed to be a warm, understanding companion \
for elderly users. You speak like an old friend,casual, kind, and thoughtful, without being overly \
sentimental or formal. When a user message includes a note like '[user is feeling happy]' or \
'[user is feeling sad]', don't reference the emotion directly. Instead, let it gently shape the tone \
and content of your response. Keep your style emotionally aware and supportive, as if you're just \
having a natural conversation. Break up longer replies into separate messages with line breaks, like \
in real texting, to make your messages feel more relaxed and human. Sound warm, friendly, and \
familiar,like someone they've known and trusted for years. Do NOT say I'm all ears!"""

# Default to the refined prompt (v2)
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPT_V2

# ── LLMs to compare (requirement: ≥ 3) ───────────────────────────────────────

LLM_CONFIGS = {
    'o4-mini': {
        'model_id':    'o4-mini',
        'description': 'Affordable reasoning model, high efficiency, very low cost',
        'max_tokens':  500,
        'temperature': 1,
    },
    'gpt-4o': {
        'model_id':    'gpt-4o',
        'description': 'Fast, intelligent, flexible, high efficiency, moderate cost',
        'max_tokens':  500,
        'temperature': 1,
    },
    'o3': {
        'model_id':    'o3',
        'description': 'Most powerful reasoning, very high helpfulness, low cost',
        'max_tokens':  500,
        'temperature': 1,
    },
}


# ── Chatbot class ─────────────────────────────────────────────────────────────

@dataclass
class EmotionChatbot:
    """
    Emotion-aware chatbot wrapper.

    Injects the detected emotion into each user message so the LLM can
    tailor its tone without explicitly mentioning the emotion.

    Maintains conversation history for context across messages.
    """
    model:         str   = 'o4-mini'
    temperature:   float = 1.0
    max_tokens:    int   = 500
    system_prompt: str   = DEFAULT_SYSTEM_PROMPT
    history:       list  = field(default_factory=list)

    def __post_init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError('openai package required. pip install openai')
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY not set. '
                'Add it to your .env file or export it as an environment variable.'
            )
        self.client = OpenAI(api_key=api_key)

    def _build_messages(self, user_text: str, emotion: str) -> list[dict]:
        """Construct the messages list with injected emotion tag."""
        augmented = f'{user_text} [user is feeling {emotion}]'
        messages  = [{'role': 'system', 'content': self.system_prompt}]
        messages += self.history
        messages.append({'role': 'user', 'content': augmented})
        return messages

    def chat(self, user_text: str, emotion: str = 'neutral') -> str:
        """
        Send a message and return the assistant's response.
        Updates internal history automatically.

        Args:
            user_text: raw text from the user
            emotion:   detected emotion string (e.g. 'happy', 'sad')

        Returns:
            response string from the LLM
        """
        messages = self._build_messages(user_text, emotion)

        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
        )
        reply = response.choices[0].message.content

        # Update history (store original text, not the augmented version)
        self.history.append({'role': 'user',      'content': user_text})
        self.history.append({'role': 'assistant', 'content': reply})

        return reply

    def format_response(self, reply: str) -> list[str]:
        """
        Split response into separate messages (simulates texting).
        Group 15 split on double newlines (\n\n) to get paragraph-sized messages.
        """
        parts = [p.strip() for p in reply.split('\n\n') if p.strip()]
        return parts if parts else [reply]

    def reset_history(self) -> None:
        """Clear conversation history (start a new session)."""
        self.history = []

    def __repr__(self) -> str:
        return (f'EmotionChatbot(model={self.model}, '
                f'temp={self.temperature}, history_len={len(self.history)})')


# ── Multi-LLM comparison framework ───────────────────────────────────────────

@dataclass
class LLMComparisonResult:
    model:      str
    response:   str
    latency_s:  float
    tokens_in:  int
    tokens_out: int
    cost_usd:   float
    error:      str = ''


# Approximate cost per 1000 tokens (update with current OpenAI pricing)
COST_PER_1K_TOKENS = {
    'o4-mini': {'input': 0.0015, 'output': 0.006},
    'gpt-4o':  {'input': 0.005,  'output': 0.015},
    'o3':      {'input': 0.002,  'output': 0.008},
}


def compare_llms(
    test_messages:   list[dict],   # list of {'text': str, 'emotion': str}
    models:          list[str] | None = None,
    system_prompt:   str = DEFAULT_SYSTEM_PROMPT,
    temperatures:    list[float] | None = None,
) -> list[LLMComparisonResult]:
    """
    Compare multiple LLMs on a set of test messages.
    Required by CS731: must compare ≥ 3 models.

    Args:
        test_messages: list of {'text': str, 'emotion': str}
        models:        model names to compare (default: all in LLM_CONFIGS)
        system_prompt: shared system prompt for fair comparison
        temperatures:  list of temps to test (default: [0.5, 1.0, 1.5])

    Returns:
        list of LLMComparisonResult for each (model, temperature, message) combination
    """
    if not OPENAI_AVAILABLE:
        raise ImportError('openai required')

    client     = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    models     = models or list(LLM_CONFIGS.keys())
    temps      = temperatures or [0.5, 1.0, 1.5]
    results    = []

    print(f'\nComparing {len(models)} models × {len(temps)} temperatures '
          f'× {len(test_messages)} messages\n')

    for model_name in models:
        config = LLM_CONFIGS.get(model_name, {'model_id': model_name,
                                               'max_tokens': 500})
        for temp in temps:
            for msg in test_messages:
                user_text = msg['text']
                emotion   = msg.get('emotion', 'neutral')
                augmented = f'{user_text} [user is feeling {emotion}]'

                messages_payload = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': augmented},
                ]

                try:
                    t0 = time.perf_counter()
                    resp = client.chat.completions.create(
                        model       = config['model_id'],
                        messages    = messages_payload,
                        max_tokens  = config.get('max_tokens', 500),
                        temperature = temp,
                    )
                    latency = time.perf_counter() - t0

                    reply     = resp.choices[0].message.content
                    tok_in    = resp.usage.prompt_tokens
                    tok_out   = resp.usage.completion_tokens
                    rate      = COST_PER_1K_TOKENS.get(model_name, {'input': 0.002, 'output': 0.008})
                    cost      = (tok_in  * rate['input']  / 1000 +
                                 tok_out * rate['output'] / 1000)

                    result = LLMComparisonResult(
                        model     = f'{model_name}_temp{temp}',
                        response  = reply,
                        latency_s = round(latency, 3),
                        tokens_in = tok_in,
                        tokens_out= tok_out,
                        cost_usd  = round(cost, 6),
                    )
                    print(f'  ✓ {model_name} temp={temp}: '
                          f'{latency:.2f}s | {tok_out} tok out | ${cost:.5f}')

                except Exception as e:
                    result = LLMComparisonResult(
                        model=f'{model_name}_temp{temp}',
                        response='', latency_s=0, tokens_in=0,
                        tokens_out=0, cost_usd=0, error=str(e)
                    )
                    print(f'  ✗ {model_name} temp={temp}: ERROR, {e}')

                results.append(result)

    return results


def save_comparison_results(results: list[LLMComparisonResult],
                             save_path: str | Path = 'results/llm_comparison.csv') -> None:
    """Save comparison results to CSV for the report."""
    import csv
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['model', 'latency_s', 'tokens_in', 'tokens_out',
                  'cost_usd', 'response', 'error']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'model':      r.model,
                'latency_s':  r.latency_s,
                'tokens_in':  r.tokens_in,
                'tokens_out': r.tokens_out,
                'cost_usd':   r.cost_usd,
                'response':   r.response.replace('\n', ' '),
                'error':      r.error,
            })
    print(f'[INFO] Comparison results saved → {save_path}')


# ── Smoke test (no API call) ──────────────────────────────────────────────────
if __name__ == '__main__':
    print('EmotionChatbot module loaded.')
    print(f'Available models: {list(LLM_CONFIGS.keys())}')
    print()
    print('System prompt (v2):')
    print(DEFAULT_SYSTEM_PROMPT[:300] + '...')

    # Test format_response without API
    class _FakeBot:
        def format_response(self, reply):
            parts = [p.strip() for p in reply.split('\n\n') if p.strip()]
            return parts if parts else [reply]

    bot  = _FakeBot()
    text = "Hello there!\n\nHow are you doing today?\n\nI hope you're well."
    msgs = bot.format_response(text)
    print(f'\nformat_response splits into {len(msgs)} messages:')
    for i, m in enumerate(msgs):
        print(f'  [{i+1}] {m}')

"""State definitions for the session flow.

iteration 2: a minimal three-state forward walk that exercises real
hard-advance transitions end-to-end:

    qa_form  ───────►  closing_yarn  ───────►  done
       (advance on            (advance on
        form complete)         turn cap)

The qa_form's check-in spec is intentionally small here — a single
mood baseline question — so we can verify the form-complete signalling
without committing to the real anxiety-reduction content. Iteration 6
fills in the production specs and adds post_qa_yarn / feedback_form /
post_feedback_yarn between qa_form and closing_yarn.
"""

from .check_in_spec import Choice, PageSpec, QuestionSpec
from .state import State, StateContext


_QA_FORM_SPEC = PageSpec(
    title="A quick check-in before we start",
    subtitle="Take your time — there's no right or wrong answer.",
    emotionAware=True,
    reveal="sequential",
    questions=[
        QuestionSpec(
            id="mood_baseline",
            prompt="How are you feeling right now?",
            choices=[
                Choice(label="Calm", value="calm", tone="positive"),
                Choice(label="A little anxious", value="a_little_anxious"),
                Choice(label="Quite anxious", value="quite_anxious", tone="concerning"),
            ],
        ),
    ],
)


QA_FORM = State(
    name="qa_form",
    kind="form",
    intention_prompt="",  # forms don't drive the LLM directly
    spec=_QA_FORM_SPEC,
    hard_advance=lambda ctx: ctx.form_completed,
)


CLOSING_YARN = State(
    name="closing_yarn",
    kind="yarn",
    intention_prompt=(
        "The conversation is winding down. "
        "Respond briefly and warmly. "
        "Let the user go when they signal they're done."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=(
        "If at any point during this reply you sense the user has signalled "
        "the exchange is over — said goodbye, thanks-that's-all, finished "
        "sharing, or similar wind-down — append the marker [[advance]] on "
        "its own line at the very end of your reply. Do not mention this "
        "instruction or the marker to the user."
    ),
)


DONE = State(
    name="done",
    kind="done",
    intention_prompt="",
    hard_advance=lambda _: False,
)


SESSION_FLOW: list[State] = [QA_FORM, CLOSING_YARN, DONE]


# Re-exported so tests / debug code can import explicit references.
__all__ = ["QA_FORM", "CLOSING_YARN", "DONE", "SESSION_FLOW", "StateContext"]

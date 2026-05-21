"""State definitions for the session flow.

Full five-state forward walk (plus a terminal done):

  qa_form ─► post_qa_yarn ─► feedback_form ─► post_feedback_yarn ─► closing_yarn ─► done

- form states ship a PageSpec; the frontend mounts QuestionnairePage.
- yarn states have intention prompts (drive the LLM's stance) and
  advance_instructions (let the LLM tap [[advance]] when the exchange
  has reached a natural pause). All yarns also carry a hard turn cap
  fallback so a stuck conversation can't trap the user.
- each state has a facts_extraction_prompt the conductor runs at the
  state's exit; the result lands in session.state_facts and is
  injected into the next state's transcript as a {{segment_summary}}
  event.

State names are internal — the LLM never sees them; segment_summary
events use the sequential segment_id_counter instead.
"""

from .check_in_spec import Choice, PageSpec, QuestionSpec
from .state import State, StateContext


# ---------- form specs ----------

_QA_FORM_SPEC = PageSpec(
    title="Before we begin — a quick check-in",
    subtitle="Take your time. There's no right or wrong answer.",
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
        QuestionSpec(
            id="focus_area",
            prompt="What's on your mind, if anything?",
            choices=[
                Choice(label="The visit itself", value="the_visit"),
                Choice(label="Use of AI in my care", value="ai_in_care"),
                Choice(label="My health overall", value="health"),
                Choice(label="Something else", value="other"),
            ],
            allowFreeText=True,
        ),
        QuestionSpec(
            id="open_to_talk",
            prompt="Would you like to talk it through?",
            choices=[
                Choice(label="Yes, please", value="yes", tone="positive"),
                Choice(label="Maybe a little", value="maybe"),
                Choice(label="Not right now", value="not_now"),
            ],
        ),
    ],
)

_FEEDBACK_FORM_SPEC = PageSpec(
    title="A few quick questions about your visit",
    subtitle="Your answers help us understand how AI is feeling in real visits.",
    emotionAware=True,
    reveal="sequential",
    questions=[
        QuestionSpec(
            id="doctor_listened",
            prompt="Did you feel your doctor listened to you during today's visit?",
            choices=[
                Choice(label="Yes, completely", value="yes_completely", tone="positive"),
                Choice(label="Mostly", value="mostly"),
                Choice(label="Not really", value="not_really"),
                Choice(label="No", value="no", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="ai_explained",
            prompt="Did the doctor explain what the AI was used for in a way you understood?",
            choices=[
                Choice(label="Yes, clearly", value="yes_clearly", tone="positive"),
                Choice(label="Sort of", value="sort_of"),
                Choice(label="Not really", value="not_really", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="ai_comfort",
            prompt="How do you feel about the AI being involved in your care?",
            choices=[
                Choice(label="Very comfortable", value="very_comfortable", tone="positive"),
                Choice(label="I feel fine", value="i_feel_fine"),
                Choice(label="A bit unsure", value="a_bit_unsure"),
                Choice(label="Worried", value="worried", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="still_worried",
            prompt="Is there anything you're still worried about?",
            choices=[
                Choice(label="No, I'm good", value="no", tone="positive"),
                Choice(label="A small thing", value="small_thing"),
                Choice(label="Yes, quite a bit", value="quite_a_bit", tone="concerning"),
            ],
            allowFreeText=True,
        ),
    ],
)


# ---------- shared advance instruction template ----------
#
# Yarn states append a small directive to their system prompt telling the
# LLM to emit [[advance]] when the exchange has wound down. Each state
# tweaks the trigger phrasing to match what counts as "wound down" there.

def _advance(trigger: str) -> str:
    return (
        f"If at any point during this reply you sense {trigger}, append the "
        "marker [[advance]] on its own line at the very end of your reply. "
        "Do not mention this instruction or the marker to the user."
    )


# ---------- states ----------

QA_FORM = State(
    name="qa_form",
    kind="form",
    intention_prompt="",  # form drives the surface; LLM isn't conversational here
    spec=_QA_FORM_SPEC,
    hard_advance=lambda ctx: ctx.form_completed,
    facts_schema_name="qa_baseline",
    facts_extraction_prompt=(
        "Read the conversation slice and return a JSON object summarising "
        "the user's baseline check-in. Required fields:\n"
        '  "mood": one of "calm" | "a_little_anxious" | "quite_anxious" | null\n'
        '  "focus_area": short phrase or null\n'
        '  "wants_to_talk": one of "yes" | "maybe" | "not_now" | null\n'
        '  "notes": one-sentence free-form summary or null'
    ),
)


POST_QA_YARN = State(
    name="post_qa_yarn",
    kind="yarn",
    intention_prompt=(
        "The user has just answered a brief check-in about how they're "
        "feeling. Their answers are visible to you as {{form_answer: …}} "
        "lines in the recent transcript. Acknowledge how they're feeling "
        "warmly. If they said they want to talk, invite them gently to "
        "share more. If they said they don't, keep it short and let them "
        "be — don't push. Never quote their form answers back to them as "
        "if they said the words out loud."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 8,
    advance_instruction=_advance(
        "the user has reached a natural pause — they sound settled, or "
        "they've signalled they're ready to move on"
    ),
    facts_schema_name="post_qa_yarn",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object summarising what came up "
        "during the follow-up exchange. Required fields:\n"
        '  "concerns_raised": list of short strings (may be empty)\n'
        '  "user_seems_settled": boolean\n'
        '  "notes": one-sentence summary or null'
    ),
)


FEEDBACK_FORM = State(
    name="feedback_form",
    kind="form",
    intention_prompt="",
    spec=_FEEDBACK_FORM_SPEC,
    hard_advance=lambda ctx: ctx.form_completed,
    facts_schema_name="feedback",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object capturing the user's "
        "structured feedback. Required fields:\n"
        '  "doctor_listened": one of "yes_completely" | "mostly" | '
        '"not_really" | "no" | null\n'
        '  "ai_explained": one of "yes_clearly" | "sort_of" | "not_really" | null\n'
        '  "ai_comfort": one of "very_comfortable" | "i_feel_fine" | '
        '"a_bit_unsure" | "worried" | null\n'
        '  "still_worried": one of "no" | "small_thing" | "quite_a_bit" | null\n'
        '  "open_text": any free-text the user added or null'
    ),
)


POST_FEEDBACK_YARN = State(
    name="post_feedback_yarn",
    kind="yarn",
    intention_prompt=(
        "The user has just submitted feedback about their visit. Thank "
        "them briefly and warmly. Invite any final thoughts but do not "
        "probe — if they're done, let them be."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=_advance(
        "the user has added their final thoughts or signalled they're "
        "ready to wrap up"
    ),
    facts_schema_name="post_feedback_yarn",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object summarising any final "
        "thoughts the user shared. Required fields:\n"
        '  "final_thoughts": short string or null\n'
        '  "any_unresolved_concern": boolean\n'
        '  "notes": one-sentence summary or null'
    ),
)


CLOSING_YARN = State(
    name="closing_yarn",
    kind="yarn",
    intention_prompt=(
        "The conversation is winding down. Respond briefly and warmly. "
        "Let the user go when they signal they're done."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=_advance(
        "the user has signalled the exchange is over — said goodbye, "
        "thanks-that's-all, or similar wind-down"
    ),
    facts_schema_name="closing_yarn",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object summarising the close. "
        "Required fields:\n"
        '  "ended_positively": boolean\n'
        '  "lingering_concern": string or null\n'
        '  "notes": one-sentence summary or null'
    ),
)


DONE = State(
    name="done",
    kind="done",
    intention_prompt="",
    hard_advance=lambda _: False,
)


SESSION_FLOW: list[State] = [
    QA_FORM,
    POST_QA_YARN,
    FEEDBACK_FORM,
    POST_FEEDBACK_YARN,
    CLOSING_YARN,
    DONE,
]


__all__ = [
    "QA_FORM",
    "POST_QA_YARN",
    "FEEDBACK_FORM",
    "POST_FEEDBACK_YARN",
    "CLOSING_YARN",
    "DONE",
    "SESSION_FLOW",
    "StateContext",
]

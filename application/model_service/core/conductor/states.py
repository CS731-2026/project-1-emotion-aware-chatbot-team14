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
    title="A few quick questions about this chat",
    subtitle="Your answers help us understand how this assistant felt to talk to.",
    emotionAware=True,
    reveal="sequential",
    questions=[
        QuestionSpec(
            id="felt_heard",
            prompt="Did you feel this assistant listened to you?",
            choices=[
                Choice(label="Yes, completely", value="yes_completely", tone="positive"),
                Choice(label="Mostly", value="mostly"),
                Choice(label="Not really", value="not_really"),
                Choice(label="No", value="no", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="responses_helpful",
            prompt="Were the assistant's replies helpful to you?",
            choices=[
                Choice(label="Very helpful", value="very_helpful", tone="positive"),
                Choice(label="Somewhat helpful", value="somewhat"),
                Choice(label="Not really", value="not_really", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="comfort_using",
            prompt="How comfortable did you feel using this assistant?",
            choices=[
                Choice(label="Very comfortable", value="very_comfortable", tone="positive"),
                Choice(label="Mostly fine", value="mostly_fine"),
                Choice(label="A bit unsure", value="a_bit_unsure"),
                Choice(label="Uncomfortable", value="uncomfortable", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="would_use_again",
            prompt="Would you use this kind of assistant again?",
            choices=[
                Choice(label="Yes, definitely", value="yes_definitely", tone="positive"),
                Choice(label="Maybe", value="maybe"),
                Choice(label="Probably not", value="probably_not", tone="concerning"),
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
    # Safety net only — preferred exit is the [[advance]] emission below,
    # fired by the LLM when it senses the user is ready to move on. The
    # turn / time caps stop a stuck conversation from trapping the user.
    hard_advance=lambda ctx: ctx.turn_in_state >= 24 or ctx.elapsed_in_state >= 900.0,
    advance_instruction=_advance(
        "the user has reached a natural pause — they sound settled, or "
        "they've signalled they're ready to move on, or they've agreed "
        "to share feedback about this assistant"
    ),
    # After a couple of warm-up turns, start gently steering toward the
    # feedback hand-off. Soft language only — never pressure the user.
    # The LLM exits via [[advance]] when the user agrees (see the
    # advance_instruction above).
    late_guidance_after=2,
    late_guidance=(
        "By now the user has had a couple of turns to settle. Begin to "
        "gently steer the conversation toward sharing brief feedback "
        "about this assistant — only when it feels natural and the user "
        "seems ready. Phrase it as a soft invitation ('if you're up for "
        "it, I'd love to hear how this felt'), not a demand. If they "
        "decline or deflect, let it go and follow their lead. Do not "
        "mention 'feedback form' or any UI vocabulary — just frame it "
        "as sharing their thoughts."
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
        "structured feedback about this assistant. Required fields:\n"
        '  "felt_heard": one of "yes_completely" | "mostly" | '
        '"not_really" | "no" | null\n'
        '  "responses_helpful": one of "very_helpful" | "somewhat" | "not_really" | null\n'
        '  "comfort_using": one of "very_comfortable" | "mostly_fine" | '
        '"a_bit_unsure" | "uncomfortable" | null\n'
        '  "would_use_again": one of "yes_definitely" | "maybe" | "probably_not" | null\n'
        '  "open_text": any free-text the user added or null'
    ),
)


POST_FEEDBACK_YARN = State(
    name="post_feedback_yarn",
    kind="yarn",
    intention_prompt=(
        "The user has just submitted feedback about this assistant. "
        "Thank them briefly and warmly. Invite any final thoughts but "
        "do not probe — if they're done, let them be."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=_advance(
        "the user has added their final thoughts or signalled they're "
        "ready to wrap up"
    ),
    # One turn of warm acknowledgement; on the second turn, start
    # easing toward a close so the conversation doesn't drag.
    late_guidance_after=1,
    late_guidance=(
        "The user has had one turn to add any final thoughts. If they "
        "haven't volunteered anything substantial, gently move toward a "
        "warm close — acknowledge they've shared, thank them, and leave "
        "an opening to say goodbye. Don't extract more if they seem "
        "done. Do not mention any UI or system vocabulary."
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

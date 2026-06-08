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

State names are internal, the LLM never sees them; segment_summary
events use the sequential segment_id_counter instead.
"""

from .check_in_spec import Choice, PageSpec, QuestionSpec
from .state import State, StateContext, TickResult


# ---------- form specs ----------
#
# Questions sourced from design_decision/gp_feedback_questions.html.
# Scales (1–10, 1–5) are collapsed into three labelled tiers, the
# QuestionSpec chip model does not support numeric sliders.

_QA_FORM_SPEC = PageSpec(
    title="About your appointment today",
    subtitle="This won't take long, tap the answer that feels right.",
    emotionAware=True,
    reveal="sequential",
    questions=[
        QuestionSpec(
            id="doctor_explained",
            prompt="Did your doctor explain clearly what was wrong with you today?",
            choices=[
                Choice(label="Yes, very clearly", value="yes_clearly", tone="positive"),
                Choice(label="Mostly, yes", value="mostly"),
                Choice(label="It was a bit confusing", value="confusing", tone="concerning"),
                Choice(label="No, I'm still unsure", value="still_unsure", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="treatment_confidence",
            prompt="How confident do you feel about following the treatment or advice given today?",
            choices=[
                Choice(label="Very confident", value="very_confident", tone="positive"),
                Choice(label="Fairly confident", value="fairly_confident"),
                Choice(label="Not very confident", value="not_confident", tone="concerning"),
                Choice(label="I'm not sure what to do", value="unsure_what_to_do", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="symptom_instructions",
            prompt="Were you told what to do if your symptoms get worse or don't improve?",
            choices=[
                Choice(label="Yes", value="yes", tone="positive"),
                Choice(label="Partly", value="partly"),
                Choice(label="No", value="no", tone="concerning"),
            ],
        ),
    ],
)

_FEEDBACK_FORM_SPEC = PageSpec(
    title="About technology and your experience",
    subtitle="Your answers are anonymous and help improve care for everyone.",
    emotionAware=True,
    reveal="sequential",
    questions=[
        QuestionSpec(
            id="ai_noticed",
            prompt="Did you notice a computer or AI tool being used during your appointment?",
            choices=[
                Choice(label="Yes, I noticed", value="yes_noticed"),
                Choice(label="I think so", value="think_so"),
                Choice(label="Not sure", value="not_sure"),
                Choice(label="No", value="no"),
            ],
        ),
        QuestionSpec(
            id="ai_feeling",
            prompt="How did it make you feel knowing that a computer may have helped with your appointment?",
            choices=[
                Choice(label="Reassured, more accurate", value="reassured", tone="positive"),
                Choice(label="Fine, no strong feelings", value="fine"),
                Choice(label="A little unsure", value="a_little_unsure", tone="concerning"),
                Choice(label="Worried or uncomfortable", value="worried", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="privacy_felt_safe",
            prompt="Did you feel your personal health information was kept safe and private today?",
            choices=[
                Choice(label="Yes, completely", value="yes_completely", tone="positive"),
                Choice(label="Mostly", value="mostly"),
                Choice(label="I'm not sure", value="not_sure", tone="concerning"),
                Choice(label="I have concerns", value="have_concerns", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="overall_rating",
            prompt="Overall, how would you rate today's appointment?",
            choices=[
                Choice(label="Excellent", value="excellent", tone="positive"),
                Choice(label="Good", value="good", tone="positive"),
                Choice(label="Fair", value="fair"),
                Choice(label="Poor", value="poor", tone="concerning"),
            ],
        ),
        QuestionSpec(
            id="what_would_help",
            prompt="Is there anything that would have made today's visit better for you?",
            choices=[
                Choice(label="More time with the doctor", value="more_time"),
                Choice(label="A clearer explanation", value="clearer_explanation"),
                Choice(label="More information about the technology", value="more_tech_info"),
                Choice(label="More privacy", value="more_privacy"),
                Choice(label="Nothing, it was fine", value="nothing_fine", tone="positive"),
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
        "the patient's answers about their GP appointment. Required fields:\n"
        '  "doctor_explained": one of "yes_clearly" | "mostly" | "confusing" | "still_unsure" | null\n'
        '  "treatment_confidence": one of "very_confident" | "fairly_confident" | "not_confident" | "unsure_what_to_do" | null\n'
        '  "symptom_instructions": one of "yes" | "partly" | "no" | null\n'
        '  "has_concern": boolean, true if any answer signals confusion or lack of confidence\n'
        '  "notes": one-sentence free-form summary or null'
    ),
)


class PostQaYarn(State):
    """The check-in follow-up yarn, three-phase prompt.

    Phases (keyed off `ctx.turn_in_state` inside the state):
      - 0 … 1  warm acknowledgement only (base intention_prompt)
      - 2      branching guidance: probe a bad experience, OR pivot to a
               soft ask for feedback
      - 3 +    escalate, insist on feedback, warmly but explicitly. If
               the bad-experience probe is still live, frame the
               feedback as the way to make sure their experience is
               heard.

    Subclassed so the phase logic lives next to its phrasing instead of
    being squeezed into a single `late_guidance` field. The conductor's
    default `should_advance` still applies; exit happens when the LLM
    emits `[[advance]]` (see advance_instruction) or the hard cap fires.
    """

    _BASE_INTENTION = (
        "The patient has just answered a few questions about their GP "
        "appointment today. Their answers are visible to you as "
        "{{form_answer: …}} lines in the recent transcript. "
        "Respond in plain, warm, everyday English, no medical or "
        "technical jargon. Keep your reply under three sentences. "
        "If their answers suggest confusion, worry, or lack of confidence "
        "about the visit or about AI being used, open with empathy and "
        "validate that before anything else. "
        "If they seemed fine with the appointment, acknowledge that warmly "
        "and let them know their feedback is appreciated. "
        "Never quote their form answers back to them word for word."
    )

    _BRANCHING_GUIDANCE = (
        "The patient has had a turn or two to settle. Read where they are "
        "now, then take ONE of these two paths:\n"
        "\n"
        "  (A) If they seem confused about their diagnosis, worried about "
        "AI or technology in their care, or uneasy about privacy, your "
        "priority is to understand what specifically feels wrong. Ask one "
        "gentle, plain-English question. Do not move on until they have "
        "named their concern. Once they have, reassure them simply and "
        "directly, then invite them to share that concern as feedback so "
        "it can be acted on.\n"
        "\n"
        "  (B) If they seem calm, satisfied, or neutral, invite them "
        "gently to share how the appointment felt overall. Frame it as "
        "something that helps improve care for others ('it only takes a "
        "moment and it genuinely helps'). Don't push more than twice.\n"
        "\n"
        "In both paths, never use words like 'feedback form', 'survey', "
        "'algorithm', or 'AI', say 'computer' or 'technology' if needed."
    )

    _INSIST_GUIDANCE = (
        "ESCALATION: this is at least the third turn. Be direct but still "
        "kind. Say clearly that you'd really like to hear how their visit "
        "went before they go. If they have an unresolved concern about AI "
        "or privacy, name it gently and explain they have every right to "
        "know what was used in their care. If they decline firmly, accept "
        "it gracefully and emit [[advance]] next turn."
    )

    def tick(self, ctx: StateContext) -> TickResult:
        """Phase the intention by turn count.

        Three phases, see class docstring. `should_advance` defers to
        the base hard_advance lambda (turn cap / time cap); the soft
        [[advance]] emission path is handled by the conductor outside
        tick().
        """
        if ctx.turn_in_state >= 3:
            intention = f"{self._BASE_INTENTION}\n\n{self._BRANCHING_GUIDANCE}\n\n{self._INSIST_GUIDANCE}"
        elif ctx.turn_in_state >= 2:
            intention = f"{self._BASE_INTENTION}\n\n{self._BRANCHING_GUIDANCE}"
        else:
            intention = self._BASE_INTENTION
        return TickResult(intention=intention, advance=self.hard_advance(ctx))


POST_QA_YARN = PostQaYarn(
    name="post_qa_yarn",
    kind="yarn",
    # intention_prompt is overridden by the subclass's phase logic; we
    # still set it so the yarn-opener path (which reads intention_prompt
    # directly for its turn-0 read) sees the base text.
    intention_prompt=PostQaYarn._BASE_INTENTION,
    # Safety net only, preferred exit is the [[advance]] emission below,
    # fired by the LLM when it senses the user is ready to move on. The
    # turn / time caps stop a stuck conversation from trapping the user.
    hard_advance=lambda ctx: ctx.turn_in_state >= 24 or ctx.elapsed_in_state >= 900.0,
    advance_instruction=_advance(
        "the patient has reached a natural pause, they sound settled, "
        "their concern has been acknowledged, they've agreed to share "
        "more thoughts, or they've firmly said they're done"
    ),
    facts_schema_name="post_qa_yarn",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object summarising what came up "
        "in the follow-up conversation after the appointment questions. Required fields:\n"
        '  "concerns_raised": list of short strings describing any worries '
        'the patient mentioned (may be empty)\n'
        '  "ai_concern": boolean, true if the patient expressed worry about AI or technology\n'
        '  "privacy_concern": boolean, true if the patient raised a data or privacy worry\n'
        '  "patient_seems_settled": boolean\n'
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
        "Read the slice and return a JSON object capturing the patient's "
        "structured feedback about their GP appointment and AI use. Required fields:\n"
        '  "ai_noticed": one of "yes_noticed" | "think_so" | "not_sure" | "no" | null\n'
        '  "ai_feeling": one of "reassured" | "fine" | "a_little_unsure" | "worried" | null\n'
        '  "privacy_felt_safe": one of "yes_completely" | "mostly" | "not_sure" | "have_concerns" | null\n'
        '  "overall_rating": one of "excellent" | "good" | "fair" | "poor" | null\n'
        '  "what_would_help": string or null\n'
        '  "open_text": any free-text the patient added or null\n'
        '  "has_concern": boolean, true if any answer signals worry about AI, privacy, or a poor rating'
    ),
)


POST_FEEDBACK_YARN = State(
    name="post_feedback_yarn",
    kind="yarn",
    intention_prompt=(
        "The patient has just answered questions about their appointment "
        "and about technology being used in their care. "
        "Thank them briefly and warmly in plain English, no jargon. "
        "If their answers suggest they felt worried or unsure about AI "
        "or privacy, address that directly in one or two calm sentences "
        "before thanking them. Keep it under three sentences total. "
        "Invite any final question they may have, but don't push."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=_advance(
        "the patient has asked their final question or signalled they're "
        "ready to go"
    ),
    late_guidance_after=1,
    late_guidance=(
        "The patient has had one turn. If they haven't raised a new "
        "concern, ease gently toward a warm close, thank them, say "
        "their answers help improve care, and leave an opening for "
        "goodbye. Do not repeat reassurances already given. "
        "Never use words like 'algorithm', 'data', or 'AI system'."
    ),
    facts_schema_name="post_feedback_yarn",
    facts_extraction_prompt=(
        "Read the slice and return a JSON object summarising any final "
        "concerns or questions the patient raised. Required fields:\n"
        '  "final_question": short string or null\n'
        '  "ai_concern_resolved": boolean\n'
        '  "any_unresolved_concern": boolean\n'
        '  "notes": one-sentence summary or null'
    ),
)


CLOSING_YARN = State(
    name="closing_yarn",
    kind="yarn",
    intention_prompt=(
        "The conversation is wrapping up. Say goodbye warmly and briefly "
        "in plain everyday English. Remind them their answers are "
        "anonymous and will help improve care. Let them go as soon as "
        "they're ready."
    ),
    hard_advance=lambda ctx: ctx.turn_in_state >= 4,
    advance_instruction=_advance(
        "the patient has said goodbye or signalled they're done"
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

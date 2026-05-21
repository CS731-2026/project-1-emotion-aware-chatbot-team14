"""Session conductor package.

Owns the per-session state machine and the contract between the chat router
and the reasoning agent for what to render and what intention the LLM
should be operating under.

The conductor itself never calls the LLM — it consumes deterministic
signals (turn counts, form completion, parsed [[advance]] emissions) and
walks a fixed forward-only list of states.
"""

from .conductor import Conductor, ConductorDecision
from .state import State, StateContext

__all__ = ["State", "StateContext", "Conductor", "ConductorDecision"]

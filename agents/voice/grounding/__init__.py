"""What a tool call may write, and what the model is told when it may not.

Split from realtime_worker 2026-08-26, verbatim. _handle_tool_call alone is 625
lines and was the largest function in the package.

- The set is a transitive CLOSURE, not a tidy list. is_hold_request,
  _objective_of and _hint_vocabulary are not "grounding" by any reading; they
  are here because the rejection path calls them, and leaving them behind would
  have made this module import the worker back.
- _create_response lives here for the same reason: its six call sites are split
  between this module and the worker, and it had to sit where both could import
  it. It is not a grounding concern. A turns module imports it rather than
  declaring a second one - the per-site policy in its docstring is load-bearing.
- RealtimeSession stays in the worker. Moving a class both sides mutate is how
  a cycle gets built.
- Checks that answer "did the caller say this?" stay in evidence.py and are
  imported. Nothing here re-implements one.

THE PACKAGE IS A PROXY FIRST. grounding is the busiest module in this package -
realtime_worker re-exports thirty of these names, turns.py reaches for seven
more, session and lifecycle one each - so the split was done in two steps with
the suite green between them: everything moved behind this file, which
re-exports the same surface, and only then was it sliced apart. Not one import
site outside this package changed, and none should have to.

EXPLICIT, NOT `import *`. Every name here starts with an underscore, which a
star import skips by default; and the suite's __all__ scan reads this file's
ImportFrom nodes to learn what the package re-exports, so a star would leave it
guarding nothing. That check went silently vacuous once already when evidence
became a package - see the rglob block in test_realtime_protocol.py.
"""
from agents.voice.grounding.vocabulary import (
    _CALLER_WILL_ACT,
    _CALL_SHAPE_EXITS,
    _CHOICE_SAVE_TOOLS,
    _CLAIMS_SAVED,
    _FACTUAL_ESCALATIONS,
    _HOLD_REQUEST,
    _IDENTITY_ASK,
    _MAX_SAVE_REJECTIONS,
    _ORG_WORD,
    _RETIRED_VOCAB_TEXT,
    _SELF_ID,
    _SELF_ID_WEAK,
    _STREET_ADDRESS,
    _STREET_SUFFIX,
    _agent_stalled,
    _detail_left_bare,
    _gave_name_and_dob,
    _gave_own_detail,
    _said_not_a_patient,
    _announced_an_ask,
    _claims_saved,
    _hint_vocabulary,
    _is_bare_hint_word,
    _spoken_farewell,
    closing_directive,
    is_hold_request,
)
from agents.voice.grounding.handlers import (
    _address_dropped,
    _address_offered,
    _candidate_location,
    _discarded_location,
    _strip_ungrounded_detail,
    _ungrounded_escalation,
    hospital_mismatch,
)
from agents.voice.grounding.telemetry import (
    _collected_pairs,
    _objective_of,
)
from agents.voice.grounding.teardown import (
    _create_response,
    _resolve_deferred_save,
)
from agents.voice.grounding.dispatcher import (
    _ToolOutcome,
    _handle_tool_call,
)

__all__ = [
    "_CALLER_WILL_ACT",
    "_CALL_SHAPE_EXITS",
    "_CHOICE_SAVE_TOOLS",
    "_CLAIMS_SAVED",
    "_FACTUAL_ESCALATIONS",
    "_HOLD_REQUEST",
    "_IDENTITY_ASK",
    "_MAX_SAVE_REJECTIONS",
    "_ORG_WORD",
    "_RETIRED_VOCAB_TEXT",
    "_SELF_ID",
    "_SELF_ID_WEAK",
    "_STREET_ADDRESS",
    "_STREET_SUFFIX",
    "_ToolOutcome",
    "_address_dropped",
    "_address_offered",
    "_candidate_location",
    "_claims_saved",
    "_agent_stalled",
    "_detail_left_bare",
    "_gave_name_and_dob",
    "_gave_own_detail",
    "_said_not_a_patient",
    "_announced_an_ask",
    "_spoken_farewell",
    "closing_directive",
    "_create_response",
    "_discarded_location",
    "_handle_tool_call",
    "_hint_vocabulary",
    "_is_bare_hint_word",
    "_collected_pairs",
    "_objective_of",
    "_resolve_deferred_save",
    "_strip_ungrounded_detail",
    "_ungrounded_escalation",
    "hospital_mismatch",
    "is_hold_request",
]

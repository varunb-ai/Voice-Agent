"""What the caller actually said, and whether a value is supported by it.

Every guard in here answers one question — *is there evidence for this?* — over
`sess.turns` and nothing else. None of it touches the WebSocket, the audio
loop, or the session's lifecycle; the only thing it wants from a session is the
transcript and a place to record what it refused.

SPLIT OUT OF realtime_worker.py, which had reached 7,787 lines. Not for
tidiness: pyright stops analysing a function once it is large enough, and when
it gives up it can no longer prove any local inside is read — the editor greys
out dozens of names and stops seeing the calls the module makes. That has bitten
this project before (see _handle_tool_call's docstring, and the week where every
recurring bug lived in the one function pyright had abandoned). This module is
the half that can be reasoned about on its own, so it is the half that moves.

The dependency runs ONE WAY: realtime_worker imports this, never the reverse.
That is checked, not hoped for — the extraction was chosen as the transitive
closure of these guards, and that closure contains no class and nothing from the
transport surface. Keep it that way. If something here starts needing a
RealtimeSession's behaviour rather than its transcript, it belongs on the other
side of the line.

Names are re-exported from realtime_worker so existing callers, and the suite's
`rw._thing` references, keep working unchanged.
"""
from agents.voice.evidence.patterns import (
    _ACK_TAKES_VALUE,
    _CONFIRMS_VALUE,
    _DETAIL_FUNCTION_WORDS,
    _INVITATION,
    _LOCATION_ANCHORS,
    _LOW_AUDIO_RMS,
    _MAX_OWED_PER_CALL,
    _MAX_OWED_PER_TEXT,
    _MEANING_CLASSES,
    _MIN_TURNS_FOR_ADAPTIVE,
    _NAMED_DOCTOR,
    _NON_PLACE,
    _NOT_AN_ASK,
    _NUMBER_WORD_VALUE,
    _ORG_STOPWORDS,
    _POSSESSIVE,
    _QUIET_FRACTION,
    _REPORTS_FAILURE,
    _UNGROUNDED_STOPWORDS,
    _VETTING_OPENER,
)
from agents.voice.evidence.probes import (
    _caller_ends_call,
    _class_present,
    _collapse,
    _distinctive,
    _drop_lost_substance,
    _grounded_in,
    _grounded_loosely,
    _invites_continuation,
    is_hard_refusal,
    _is_ask_for,
    _is_hint_echo,
    _is_location_ask,
    _meaning_class,
    _owed_key,
    _stem,
)
from agents.voice.evidence.window import (
    _asserted_caller_text,
    _caller_is_vetting,
    _caller_speech_level,
    _ever_transcribed,
    _other_field_probes,
    _transcript_pending,
    _turn_asserts,
)
from agents.voice.evidence.names import (
    _name_mismatch,
    _our_surname,
    _spell_out,
    _spelled_out,
    _surnames_named,
    _note_name_heard,
    _wrong_doctor_named,
)
from agents.voice.evidence.guards import (
    _grounding_verdict,
    _owed_refusal,
    _revisit_grounding,
    _rode_along,
    _ungrounded_choice,
    _ungrounded_detail,
    _ungrounded_terms,
)

__all__ = [
    "_ACK_TAKES_VALUE",
    "_CONFIRMS_VALUE",
    "_DETAIL_FUNCTION_WORDS",
    "_INVITATION",
    "_LOCATION_ANCHORS",
    "_LOW_AUDIO_RMS",
    "_MAX_OWED_PER_CALL",
    "_MAX_OWED_PER_TEXT",
    "_MEANING_CLASSES",
    "_MIN_TURNS_FOR_ADAPTIVE",
    "_NAMED_DOCTOR",
    "_NON_PLACE",
    "_NOT_AN_ASK",
    "_NUMBER_WORD_VALUE",
    "_ORG_STOPWORDS",
    "_POSSESSIVE",
    "_QUIET_FRACTION",
    "_REPORTS_FAILURE",
    "_UNGROUNDED_STOPWORDS",
    "_VETTING_OPENER",
    "_asserted_caller_text",
    "_transcript_pending",
    "_caller_ends_call",
    "_caller_is_vetting",
    "_caller_speech_level",
    "_class_present",
    "_collapse",
    "_distinctive",
    "_drop_lost_substance",
    "_ever_transcribed",
    "_grounded_in",
    "_grounded_loosely",
    "_grounding_verdict",
    "_invites_continuation",
    "is_hard_refusal",
    "_is_ask_for",
    "_is_hint_echo",
    "_is_location_ask",
    "_meaning_class",
    "_name_mismatch",
    "_our_surname",
    "_owed_key",
    "_owed_refusal",
    "_revisit_grounding",
    "_rode_along",
    "_spell_out",
    "_spelled_out",
    "_stem",
    "_surnames_named",
    "_turn_asserts",
    "_other_field_probes",
    "_ungrounded_choice",
    "_ungrounded_detail",
    "_ungrounded_terms",
    "_note_name_heard",
    "_wrong_doctor_named",
]

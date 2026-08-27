"""The pre-realtime pipeline: VAD -> Whisper -> VoiceBrain -> Piper.

DELIBERATELY EMPTY OF IMPORTS. This used to re-export VoiceBrain and CallMemory
(it is agents/voice/__init__.py, moved here on 2026-08-26), and after the move
that re-export closed a cycle:

    agents.voice.tools -> agents.experiment.memory -> THIS FILE
                       -> agents.experiment.brain  -> agents.voice.tools

which raises ImportError on a partially initialised module. It only stayed
hidden because the realtime entry points happen to import agents.voice.tools
first; importing any agents.voice module that reaches tools EARLIER — which a
new module can do without knowing it — hits it immediately.

Nothing imports VoiceBrain or CallMemory from this package root (checked), so
the re-export bought nothing and cost a cycle. Import the submodules directly.
"""

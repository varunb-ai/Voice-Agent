"""The vocabulary: every pattern, set and threshold, and nothing that acts.

PURE DATA, AND THAT IS THE RULE. Nothing here imports from the rest of this
package and nothing here calls anything. It is the bottom of the layering, so
a module that needs a word list can take it without dragging a guard along,
and a new pattern cannot quietly acquire a dependency on call state.

The comments are the point. Almost every alternative in these patterns was
added after a live call went wrong, and the call is named where it is known.
Reading this file is reading what callers have actually said.
"""
import re

_LOW_AUDIO_RMS = 0.015



# Turns that MENTION the location without asking for it: acknowledging a value
# just given, or signing off. Everything else that names a location is a request,
# whatever shape it takes.
_NOT_AN_ASK = re.compile(
    r"\b(thanks|thank you|got it|perfect|great|appreciate|have a (good|great)|"
    r"take care|goodbye|bye now|i'?ll (note|record|pass)|i have that|"
    r"that'?s all|no (problem|worries))\b", re.I)



# An acknowledgement together with the location noun it takes as its OBJECT.
#
# _NOT_AN_ASK strips the acknowledgement WORD and leaves the noun it governs,
# so "Thanks for the location." became " for the location." — still a location
# noun, still counted as an ask. Observed on call-20260820-1915: seven
# location_asks against a limit of four, and the verbatim-ask nudge firing to
# tell the agent to "stop stapling it on" about a sentence that asks nothing.
# It cost nothing that call — holds had reset the budget — but an inflated
# count ends a call early on a call without holds.
#
# The distinguishing feature is grammatical, not vocabulary: in the failing
# family the noun is the acknowledgement's object, not part of a fresh request.
# So consume the phrase whole, before the residue test runs.
#
# THE NEGATIVE LOOKAHEAD IS LOAD-BEARING. Without it the two-word gap jumps a
# clause boundary — "Great — and which campus is that" had "and which" eaten
# and the real question with it, which is the expensive direction: a missed ask
# lets the agent pester someone. Words that open a new clause end the object.
_ACK_TAKES_VALUE = re.compile(
    r"\b(thanks|thank you|appreciate|got it|perfect|great)\b"
    r"[,\s—\-]*"
    r"(?:for|on|about)?[,\s]*"
    r"(?:the|that|this|your|those)?\s*"
    r"(?:(?!(?:and|but|so|which|what|where|who|if|when|still|need)\b)\w+\s+){0,2}"
    r"(?:branch|location|office|campus|site|address)\b", re.I)



# Reading back a value the caller already gave.
# READING A VALUE BACK IS NOT ASKING FOR ONE, and the list has to cover the
# agent QUOTING the caller as well as the agent filing the value. On
# call-20260824-2014 the agent said "I heard you say she's taking the new
# patients." — a read-back by any reading — and because that phrasing was
# missing here it scored as an ASK. The grounding anchor moved past every
# caller turn that had answered, the evidence window emptied, and the guard
# stood down and accepted a status it had refused three times. The agent talked
# its own claim into the record.
_CONFIRMS_VALUE = re.compile(
    # PRESENT PROGRESSIVE TOO. The list held "i'll note" and missed "I'm just
    # noting Riverside Campus now" on call-20260827-1010 - the same act, filed
    # as it is said. It matched LOCATION_NOUN on the value it was reading back
    # and scored as a branch ask.
    r"\b(i have that as|i'?ve got that|i'?ll note|i'?m (just )?(noting|"
    r"recording|writing (that|it|this) down)|i'?ve noted|noted as|recorded as|"
    r"i'?ll put (that|it) down|so that'?s|i heard you say|"
    r"you said|what i heard|let me read (that|it) back|"
    # NOT a bare "to confirm". "I'm trying to confirm which branch she works
    # out of" is an ASK, and swallowing it would stop the budget counting the
    # commonest phrasing the agent has. The read-back sense always quotes
    # THEM — "I heard you say", "you said" — and that is the load-bearing part.
    r"i'?ll record (that|it))\b", re.I)



# Reporting that the location was NOT obtained. Names a location noun and reads
# as an ask to the inverted detector, but it is the opposite — it is the agent
# giving up. On call-20260818-1338 "I wasn't able to get the specific branch
# today" was counted as an ask, so a closing line spent a slot of the ask
# budget. Only checked on statements: "I couldn't find the branch — do you know
# it?" carries a question mark and is a genuine ask.
_REPORTS_FAILURE = re.compile(
    r"\b(was ?n'?t able|were ?n'?t able|was not able|could ?n'?t|could not|"
    r"can'?t|cannot|unable|did ?n'?t manage|no luck)\b", re.I)



# PROMISING TO ASK LATER IS NOT ASKING NOW.
#
# The third member of the family above, and the one that got onto a phone.
# call-20260827-1010: the agent said "Thanks for that - I'm just noting
# Riverside Campus now, then I'll ask about new patients." It named the topic,
# carried no question mark, and was neither a read-back nor a closing line, so
# the inverted detector scored it as an ask for the `accepting` field. Nobody
# was asked anything. Three things then ran off that phantom:
#
#   1. `_field_ask_at["accepting"]` was stamped, so the FIRST real ask forty
#      seconds later scored as a RE-ASK, and _field_already_answered went
#      looking for an answer that could only belong to another question. It
#      found "No, I don't have it." - the answer to the street address - and
#      the nudge told the model to record it as the new-patient status.
#   2. `_is_objective_ask` is the gate on the ask budget, so a turn that asked
#      for nothing spent a slot of the budget that ends the call.
#   3. It is the anchor for `_ungrounded_status`. This is the same hole
#      _CONFIRMS_VALUE was cut for on call-20260824-2014, entered from the
#      other tense: an agent turn that is ABOUT the topic while asking nothing
#      moves the evidence window past the turns that answered.
#
# CONSUMED WHOLE, like _ACK_TAKES_VALUE and for the same reason - the promise
# takes its own object, and stripping only the verb would leave the noun behind
# and change nothing. `[^.?!]*` ends the object at the sentence, so a real ask
# in the SAME turn ("...then I'll ask about new patients. Which branch?")
# survives the strip and still counts.
#
# THE DEFERRAL MARKER IS REQUIRED, and it is what keeps this narrow. A missed
# ask lets the agent pester someone, which is the expensive direction, so a
# bare "I'll ask about the branch" - which a receptionist would simply answer -
# is left alone. Only a promise that plainly points at LATER is exempt.
_DEFER = (r"(?:then|next|after (?:that|this)|afterwards?|later|"
          r"in a (?:moment|minute|second|bit)|"
          r"once (?:that'?s|we'?re|that is) (?:done|sorted|out of the way))")

_WILL_ASK = r"i(?:'?ll|'?m going to|'?m gonna| will)\s+(?:then\s+)?ask\b"

#
# `(?=[.!]|$)` IS THE RIGHT EDGE OF THE OBJECT, and it is load-bearing in the
# same way _ACK_TAKES_VALUE's negative lookahead is. `[^.?!]*` alone stops just
# short of a question mark, having already eaten the question: "Then I'll ask
# about new patients - are you taking any?" had the real ask consumed by the
# promise. So the object must END at a full stop or the end of the turn. A
# promise whose own sentence carries a question mark is not stripped at all -
# it is counted as an ask, which is the over-counting side.
_ANNOUNCES_ASK = re.compile(
    # "then I'll ask about new patients" - marker before the promise
    rf"\b{_DEFER}\b[,\s\u2014\-]*{_WILL_ASK}[^.?!]*(?=[.!]|$)"
    # "I'll ask about new patients in a moment" - marker after it
    rf"|\b{_WILL_ASK}[^.?!]*\b{_DEFER}\b[^.?!]*(?=[.!]|$)", re.I)



# The caller putting a question TO the agent instead of answering theirs.
#
# call-20260819-2121, in sixty seconds:
#   "Sorry, who's calling again?"
#   "Um, is this about a patient or something urgent?"
#   "Is this about patient related?"
#   "How can I help you?"
# Four turns, four questions, no refusal anywhere — a front desk deciding
# whether this call is safe to engage with, which is their job. The ask budget
# counted every one of them as an ask that went unanswered, hit its limit of
# four, and told the agent to escalate. The agent then hung up on "How can I
# help you?" — an open door, and the clearest invitation on the whole call.
#
# `_caller_answered_since` was the wrong instrument to lean on here: it asks
# "did they say something substantive", and a question IS substantive. It just
# is not a refusal, and the budget exists to end calls that are going nowhere,
# not calls where the other person is still working out who they are talking
# to.
#
# Matched by SHAPE, not by a phrase list. Interrogative opener, or an offer of
# help, in a turn that contains no location — an open set of wordings with a
# closed set of shapes.
_VETTING_OPENER = re.compile(
    r"^\W*(?:um+|uh+|er+|so|sorry|okay|ok|alright|yeah|well|hi|hello)?[\s,]*"
    r"(?:who|what|why|which|where|how|is|are|was|were|do|does|did|can|could|"
    r"would|will|may|might|should|sorry)\b", re.I)



# An explicit offer to keep going. Stronger than a screening question: they are
# not deciding whether to engage, they have decided and are waiting on you.
_INVITATION = re.compile(
    r"\bhow\s+(?:can|may|could)\s+i\s+(?:help|assist)\b"
    r"|\bwhat\s+can\s+i\s+(?:do|help)\b"
    r"|\bwhat\s+(?:do|did)\s+you\s+need\b"
    r"|\bwhat(?:'?s| is)\s+(?:this|it)\s+(?:regarding|about|in regard)\b"
    r"|\bgo\s+ahead\b|\bhow\s+can\s+i\s+help\b", re.I)



# A turn made of nothing but affirmative/negative tokens and punctuation.
#
# THE DISCRIMINATOR FOR A QUESTION MARK THAT IS NOT A QUESTION. The transcriber
# punctuates by intonation, and a receptionist's rising "Yes?" — confirming
# while inviting you to go on — comes back with a "?" on it. _caller_is_vetting
# then fires on the "?" alone, because its only escape hatch is a proper noun
# beside a location anchor and a one-word affirmative has nothing for it to
# find. See the CHOICE call site below for why that mattered.
#
# Deliberately NOT a general "is this interrogative" test. This matches only
# turns with no content beyond the affirmative, so "Yes, that's right?" and
# "Yeah, hi David, how are you?" are untouched — the first because it may be
# echoing our words back for confirmation, the second because it plainly is a
# question. Losing those costs one turn; accepting a real question as an answer
# is the failure _turn_asserts was built for.
_ONLY_AFFIRM = re.compile(
    r"^[\W_]*(?:(?:yes|yeah|yep|yup|no|nope|nah|sure|correct|right|speaking|"
    r"uh|um|oh|ok|okay)\b[\W_]*)+$", re.I)



# Words that anchor a location. A distinctive word sitting next to one of these
# is a candidate place name; the same word anywhere else is just a word. The
# adjacency requirement is what keeps this from firing on every proper noun in
# the call — "Hello, David" has no anchor near it.
_LOCATION_ANCHORS = frozenset({
    "branch", "branches", "campus", "campuses", "clinic", "clinics",
    "office", "offices", "center", "centre", "centers", "centres",
    "hospital", "hospitals", "location", "locations", "site", "sites",
    "building", "tower", "wing", "block", "street", "road", "avenue",
    "boulevard", "lane", "drive", "parkway", "suite", "floor", "area",
})



# Conversational words that will happily sit next to an anchor while naming no
# place at all: "the main branch", "our other office", "which location".
_NON_PLACE = frozenset({
    "main", "other", "another", "same", "this", "that", "these", "those",
    "our", "their", "his", "her", "its", "one", "two", "both", "all", "any",
    "some", "each", "every", "which", "what", "where", "when", "who", "why",
    "here", "there", "yes", "yeah", "yep", "no", "not", "but", "for", "with",
    "from", "about", "only", "just", "also", "still", "sorry", "please",
    "thanks", "thank", "hello", "hey", "okay", "sure", "right", "well",
    "you", "your", "yours", "we", "our", "they", "them", "him", "she", "he",
    "are", "was", "were", "have", "has", "had", "does", "did", "can",
    "could", "will", "would", "should", "need", "needs", "want", "know",
    "tell", "say", "said", "give", "gave", "get", "got", "see", "sees",
    "working", "works", "work", "patients", "patient", "doctor", "doctors",
    "emergency", "call", "calling", "called", "number", "details", "detail",
    "information", "anything", "something", "nothing", "everything",
    "speaking", "moment", "minute", "second", "wait", "hold", "checking",
    # Capitalisation is doing the heavy lifting, so this list only has to
    # cover words that survive it — sentence-initial ones, where every word is
    # capitalised whatever it is.
    "closed", "open", "sorry", "sure", "try", "let", "hang", "just", "look",
    "there's", "thats", "yeah", "well", "actually", "maybe", "probably",
})



# Words that carry no identifying information, so their presence in the
# transcript proves nothing about whether the caller named a real place.
_UNGROUNDED_STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "our", "their", "and",
    "branch", "branches", "office", "offices", "campus", "campuses",
    "clinic", "clinics", "center", "centre", "centers", "centres",
    "hospital", "location", "locations", "site", "sites", "medical",
    "building", "unit", "practice", "city", "street", "road", "avenue",
}



# Words that appear in almost every healthcare organisation's name. Matching on
# these would make "Methodist Medical Center" look like "Northside Medical
# Group", which is exactly the confusion this check exists to catch.
_ORG_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "for", "at", "st", "saint",
    "hospital", "hospitals", "clinic", "clinics", "medical", "medicine",
    "health", "healthcare", "center", "centre", "group", "practice",
    "associates", "physicians", "care", "services", "system", "systems",
    "institute", "department", "dept", "office", "offices", "campus",
})



# Numbers written as words, mapped to their value. The value is needed to tell
# RENDERING from SUBSTITUTION, which is the whole difficulty here:
#
#   caller "1825 4th Street"   -> "1825 Fourth Street"     rendering. Fine.
#   caller "1844th Street"     -> "eighteen forty fourth"  substitution. Not.
#
# Both replace digits with words. The first keeps a digit the caller gave and
# spells an ordinal that traces back to one ("4th" -> "fourth"); the second
# erases the number entirely and nothing in it traces anywhere. A test that
# just looked for number-words blocked both, and blocking the first throws
# away a correct address — the expensive direction.
#
# NOT a general parser. "eighteen forty fourth" is genuinely ambiguous between
# 1844th, 18 44th and 1840 4th, and picking one would be inventing an address.
# Each word is checked on its own: did the caller say this word, or the digit
# it stands for? That question has an answer without resolving the ambiguity.
#
# "a" and "an" are absent on purpose: articles far more often than quantities,
# and treating them as numbers would reject half of all real branch names.
_NUMBER_WORD_VALUE: dict[str, int] = {
    **{w: i for i, w in enumerate("""
        zero one two three four five six seven eight nine ten eleven twelve
        thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
    """.split())},
    **{w: v for w, v in zip(
        "thirty forty fifty sixty seventy eighty ninety".split(),
        range(30, 100, 10))},
    "hundred": 100, "thousand": 1000,
    **{w: i + 1 for i, w in enumerate("""
        first second third fourth fifth sixth seventh eighth ninth tenth
        eleventh twelfth thirteenth fourteenth fifteenth sixteenth
        seventeenth eighteenth nineteenth twentieth
    """.split())},
    **{w: v for w, v in zip(
        "thirtieth fortieth fiftieth sixtieth seventieth eightieth "
        "ninetieth".split(), range(30, 100, 10))},
    "hundredth": 100, "thousandth": 1000,
}



# How many times one owed sentence, and one call, may chase a recovery.
#
# NOTHING COUNTED ATTEMPTS, and on call-20260825-1435 that was a livelock. The
# mute in the delta handler is unconditional on a second spoken item; the
# recovery scheduled here is itself a response; the model produced TWO items
# for it; the second was muted, carried the same substance, and set
# `_owed_substance` again. Every pass through the loop looked exactly like the
# first, so nothing could tell it was the fourth. The caller's question was
# never answered.
#
# TWO CAPS, because there are two ways to loop and one counter sees only one of
# them. The per-text cap stops the agent chasing the identical sentence. The
# per-call cap stops the version where the model REGENERATES the owed half a
# little differently each time — same substance, different letters, a per-text
# key can never match it. Both are small on purpose: a recovery that has been
# muted twice is not being muted for a reason a third attempt fixes.
_MAX_OWED_PER_TEXT = 2



_MAX_OWED_PER_CALL = 3



# Words whose removal would change what the sentence CLAIMS rather than how it
# reads — grouped into CLASSES, and the grouping is the fix.
#
# These were a flat set, checked word by word against the transcript, and that
# made the guard fire hardest on exactly the answers the client most wants. A
# model paraphrasing the connective is the single most predictable thing it
# does:
#
#   caller "as long as they've got the right insurance"
#   model  "only if they have the right insurance"      -> EMPTIED
#   caller "they need a referral from their primary"
#   model  "only with a referral from their primary care doctor" -> EMPTIED
#
# and the same on the other side:
#
#   caller "we don't take new patients until January"
#   model  "not taking new patients until January"      -> EMPTIED
#
# In every one of those the caller DID negate, or DID make it conditional. The
# model reached for a different word for the same move. Asking whether the
# CALLER SAID "only" is the wrong question; the question is whether the caller
# expressed conditionality at all.
#
# So membership is checked per CLASS: a meaning word counts as grounded when
# ANY member of its class appears in what the caller asserted. An invented
# condition — "only if insured" on a call where nothing was conditional — still
# has no class-mate to stand on, and still drops the whole qualifier.
_MEANING_CLASSES: dict = {
    # Reversing the polarity of the claim.
    "negation": frozenset({
        "not", "never", "without", "cannot", "cant", "dont", "doesnt",
        "isnt", "arent", "wont", "wouldnt", "couldnt", "nor", "none",
        "neither", "no", "nope", "stopped", "closed", "refuse", "refused",
    }),
    # Making the claim conditional — the shape CAQH is after: "yes, but only if
    # you have insurance with this particular company". Necessity words belong
    # here too: "they need a referral" and "only with a referral" are the same
    # move, and a model will swap one for the other without hesitating.
    "condition": frozenset({
        "only", "unless", "except", "provided", "depends", "depending",
        "whether", "case", "long", "need", "needs", "needed", "require",
        "requires", "required", "must", "if", "when", "certain", "some",
    }),
}



# Auxiliaries, copulas and light verbs. Skipped outright in a QUALIFIER, the
# way _UNGROUNDED_STOPWORDS are skipped everywhere: their presence or absence
# says nothing about whether the model invented anything, and checking them
# produced "only if they the right insurance" — a sentence mangled by the
# removal of "have" because the caller had said "got".
_DETAIL_FUNCTION_WORDS = frozenset({
    "are", "is", "was", "were", "been", "being", "have", "has", "had",
    "having", "does", "did", "doing", "will", "would", "can", "could",
    "shall", "should", "get", "gets", "got", "with", "from", "their",
    "them", "they", "your", "you", "our", "its", "it", "that", "this",
    "these", "those", "there", "here", "and", "but", "for", "the", "any",
    "all", "one", "also", "just", "then", "than", "who", "which", "what",
    "about", "into", "onto", "over", "under", "been",
})



# How long a guard may wait for a caller turn that is still transcribing, and
# which tools are worth waiting for.
#
# THE MODEL HEARS AUDIO; THE GUARDS READ TRANSCRIPTS; THE TRANSCRIPT LAGS. The
# Realtime model does not wait for `input_audio_transcription` before acting —
# it works from the audio directly — so every check that reads `sess.turns` can
# be asked its question before the evidence for it exists. That is not a bug in
# any one guard; it is the two halves of this system running on different
# clocks, and it is the same cause as the record-time race _revisit_grounding
# repairs after the fact.
#
# call-20260825-1620, all three inside the same second:
#
#   16:21:37  ⚠️  'eastside' never appeared in the caller transcript
#   16:21:37  🚫 HALLUCINATED BRANCH BLOCKED: {'branch': 'Eastside Clinic'}
#   16:21:37  👤 CALLER : He's at the Eastside clinic.      <- the evidence
#
# The caller said it. The guard called it a fabrication because it asked half a
# second early, and the cost was not the record — the branch was re-saved
# correctly 30 seconds later — it was the conversation. The agent was told it
# had invented the answer, asked for a street address instead, and the next
# half-minute is barge-in wreckage: "new patients", "Actually,", "referral",
# "campus".
#
# The first answer to that was a blocking wait: hold the guard up to 1.5s for
# the words to arrive. THE MEASUREMENT IT ASKED FOR KILLED IT. That comment
# ended "`transcript_waits` records every wait so the ceiling can be set from
# measurement rather than from this comment", and across 119 call artifacts the
# distribution came back:
#
#   n=14   timeout 12   landed 0   discarded 0
#   waited_s: 1.5 1.5 1.5 1.5 1.5 1.5 1.51 1.52 1.52 1.53 1.53 1.53 1.53 1.53
#
# A spike on the ceiling and not one early return. The wait never once did its
# job, and it was not a self-inflicted deadlock either — the poll loop yielded
# every 50ms, so the reader task was free to deliver a transcript that simply
# never came that fast. It cost 1.5s of latency on every save that hit it; on
# call-20260827-1010 that is the whole of `ours 1.53s` in a 3.44s reply.
#
# So the wait is gone and the answer is entirely the DEFERRAL: a save whose
# guard objects while the words are still in flight is held, and judged on the
# transcript event itself — see _resolve_deferred_save, and the comment at that
# call site, which is the moment the wait was standing in for. The predicate
# below is what both of them asked; now only one caller is left.


# THE CALLER ASKING TO HANG UP. Deliberately narrow.
#
# call-20260826-1656 ran 193s and the last 23 of them were this, verbatim:
#
#     caller  Thank you bye Cut the call.
#     AGENT   Take care.
#     caller  How many times you will tell me bro?
#     AGENT   Right, ending here - bye.
#     caller  Bye-bye.          AGENT  Bye.
#     caller  Bye, I said.      AGENT  Goodbye.
#     caller  Hahaha.           AGENT  Take care.
#
# The model was not misbehaving - it said goodbye every single time. Nothing
# in the application could act on it, because sess.done had exactly two
# triggers and neither was reachable by the caller.
#
# EXPLICIT TOKENS ONLY, and no sentiment. "How many times you will tell me
# bro?" is the clearest statement of intent on that whole call and it is NOT
# matched here: reading frustration is a judgement, and a judgement that
# hangs up on a caller mid-sentence is the expensive direction. A farewell
# or a direct instruction is a fact.
#
# "by the way" does not match: \bbye\b requires the e.
_CALLER_ENDS_CALL = re.compile(
    r"\b(?:"
    r"bye\s*[-,]?\s*bye|bye|goodbye|good\s?bye"
    r"|cut the call|end the call|hang up|disconnect the call"
    r"|stop the call|that\'?s all,?\s*(?:thanks|thank you)"
    r")\b", re.I)



# A doctor named in speech: "Dr. Kapoor", "Doctor Smith", "Dr Okafor's".
_NAMED_DOCTOR = re.compile(r"\b(?:dr\.?|doctor)\s+([a-z][a-z'-]{2,})", re.I)



# A possessive on the end of a captured surname: "Okafor's", "Jones'".
# A SUFFIX, matched as one — `.rstrip("'s")` looks like it removes this and does
# not: rstrip takes a SET OF CHARACTERS, so it eats every trailing apostrophe
# and every trailing s. "Reyes" came back "reye".
#
# Live, on call-20260825-1625: the caller said "Dr. Reyes is an oncologist" —
# transcribed perfectly, the right doctor, the right practice — and the guard
# reported "they named 'reye', and the doctor on this call is 'reyes'". It
# refused a correct confirmation and spent a turn spelling a name nobody had
# got wrong. Every surname ending in s is affected: Reyes, Jones, Hayes,
# Brooks, Sanders, Rivers. The 1226 fixtures are Okafor, Kapoor and Smith, so
# none of them could show it.
_POSSESSIVE = re.compile(r"'s?$")



# A caller turn is "quiet" relative to how loudly THIS caller has been
# speaking, not against a fixed number. _LOW_AUDIO_RMS alone is an absolute
# threshold on a quantity that has no absolute meaning: line gain, handset,
# carrier and distance all move it, so one constant cannot be right for two
# different calls.
#
# Measured on call-20260818-1338, where the transcriber emitted "Mercy Medical
# Center" — a phrase assembled from _US_TRANSCRIBE_HINT, which names Mercy
# first among health systems and "medical center" among location words. The
# caller never said it:
#
#     real  "why are you collecting"    0.0954
#     real  "Los Angeles, California"   0.1532
#     real  "It is Los Angeles only."   0.0465
#     FAKE  "Mercy Medical Center."     0.0174     <- cleared _LOW_AUDIO_RMS (0.015)
#
# The hallucination sat just above the constant while being a quarter of this
# caller's own median level. Every fraction from 0.25 to 0.50 separates the
# four cleanly; 0.35 is the middle of that band. Checked against
# call-20260818-1112, where all four caller turns are believed genuine: none
# is flagged.
#
# RE-DERIVED 2026-08-18 against the Twilio recordings, after the accusation
# this was built on turned out to be false and after audio_rms itself was found
# to be under-reporting. Method: for each of 30 calls with a dual-channel
# recording, take the N loudest caller-channel bursts where N is the number of
# transcribed caller turns, and compute min/median over them — i.e. how quiet a
# GENUINE turn gets relative to that caller's own typical level.
#
#     lowest 0.291   p10 0.458   p25 0.662   median 0.766
#     calls with a genuine turn below median*0.35 :  2/30
#     calls with a genuine turn below median*0.20 :  0/30
#
# 0.35 was too aggressive: on ~7% of calls it would classify a real caller turn
# as quiet, and a bare one-word branch name is exactly the shape that then gets
# rejected — "'Northgate' on its own is a perfectly good answer".
#
# BE CLEAR ABOUT WHAT THIS NOW BUYS. The case it was written for (the "Mercy
# Medical Center" turn) was retracted — that audio is real. With no confirmed
# positive case and a safe calibration, the adaptive term only acts on turns
# between the absolute floor and median*0.20, which is a narrow band. It is
# kept because the reasoning still holds — an absolute constant on a
# level-dependent quantity cannot be right for two different lines — not
# because it is known to catch anything. Do not widen it without a confirmed
# fabrication to widen it against.
_QUIET_FRACTION = 0.20



# Below this many measured turns the median is not a median. One turn's
# "median" is itself, which can never be a fraction of itself, so the adaptive
# test would silently never fire — the failure mode this file keeps relearning.
_MIN_TURNS_FOR_ADAPTIVE = 3


__all__ = [
    "_ACK_TAKES_VALUE",
    "_ANNOUNCES_ASK",
    "_CALLER_ENDS_CALL",
    "_CONFIRMS_VALUE",
    "_DEFER",
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
    "_ONLY_AFFIRM",
    "_ORG_STOPWORDS",
    "_POSSESSIVE",
    "_QUIET_FRACTION",
    "_REPORTS_FAILURE",
    "_UNGROUNDED_STOPWORDS",
    "_VETTING_OPENER",
    "_WILL_ASK",
]

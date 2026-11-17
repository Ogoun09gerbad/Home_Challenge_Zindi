"""
Implementation of the "Local Agreement" policy (LA-2) used to stabilize the
streaming transcription hypotheses of a Whisper model that is re-run on
every new audio chunk.

Principle: at each iteration, Whisper re-infers the entire unconfirmed
audio buffer and produces a new hypothesis (list of words). This hypothesis
is compared with the one produced at the previous iteration on the SAME
buffer. The longest common prefix (LCP) between the two hypotheses is
considered stable: those words will no longer change and are therefore
"confirmed" (finalized). The rest (the "tail") may still be revised and
remains displayed as "partial".

Once words are confirmed, the caller (see asr_engine.py /
session_manager.py) must trim the already-consumed audio from the buffer
using the end timestamp of the last confirmed word, so that subsequent
iterations only work on genuinely new audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LocalAgreementResult:
    confirmed_words: List[str] = field(default_factory=list)
    partial_words: List[str] = field(default_factory=list)


class LocalAgreementBuffer:
    """Sliding buffer implementing the Local Agreement policy (n=2)."""

    def __init__(self) -> None:
        self._previous_hypothesis: List[str] = []

    def reset(self) -> None:
        """To be called after the audio buffer is trimmed (confirmed words)
        or after an Endpoint (new utterance)."""
        self._previous_hypothesis = []

    def update(self, new_hypothesis: List[str]) -> LocalAgreementResult:
        """Compares the new hypothesis with the previous one and returns the
        now-confirmed prefix along with the still-partial tail.
        """
        lcp_length = 0
        for prev_word, new_word in zip(self._previous_hypothesis, new_hypothesis):
            if prev_word != new_word:
                break
            lcp_length += 1

        # Never confirm the entire hypothesis at once: the last word is
        # almost always still subject to revision by upcoming context (cut
        # word, homophone, etc.), so it is always kept in the "partial"
        # part until an explicit Endpoint.
        if lcp_length >= len(new_hypothesis):
            lcp_length = max(0, len(new_hypothesis) - 1)

        result = LocalAgreementResult(
            confirmed_words=new_hypothesis[:lcp_length],
            partial_words=new_hypothesis[lcp_length:],
        )

        self._previous_hypothesis = new_hypothesis
        return result

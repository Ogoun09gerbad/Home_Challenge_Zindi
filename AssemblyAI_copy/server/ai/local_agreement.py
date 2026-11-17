"""
Implémentation de la politique de "Local Agreement" (LA-2) pour stabiliser
les hypothèses de transcription streaming d'un modèle Whisper ré-exécuté à
chaque nouveau chunk audio.

Principe : à chaque itération, Whisper ré-infère l'intégralité du buffer
audio non-confirmé et produit une nouvelle hypothèse (liste de mots). On
compare cette hypothèse à celle produite à l'itération précédente sur le
MÊME buffer. Le plus long préfixe commun (LCP) entre les deux hypothèses
est considéré comme stable : ces mots ne changeront plus et sont donc
"confirmés" (finalisés). Le reste (la "queue") est encore susceptible
d'être révisé et reste affiché comme "partiel".

Une fois des mots confirmés, l'appelant (cf. asr_engine.py /
session_manager.py) doit découper l'audio déjà consommé du buffer en
utilisant l'horodatage de fin du dernier mot confirmé, afin que les
itérations suivantes ne travaillent que sur l'audio réellement nouveau.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LocalAgreementResult:
    confirmed_words: List[str] = field(default_factory=list)
    partial_words: List[str] = field(default_factory=list)


class LocalAgreementBuffer:
    """Buffer glissant implémentant la politique Local Agreement (n=2)."""

    def __init__(self) -> None:
        self._previous_hypothesis: List[str] = []

    def reset(self) -> None:
        """À appeler après un découpage du buffer audio (mots confirmés) ou
        après un Endpoint (nouvelle phrase)."""
        self._previous_hypothesis = []

    def update(self, new_hypothesis: List[str]) -> LocalAgreementResult:
        """Compare la nouvelle hypothèse à la précédente et retourne le
        préfixe désormais confirmé ainsi que la queue encore partielle.
        """
        lcp_length = 0
        for prev_word, new_word in zip(self._previous_hypothesis, new_hypothesis):
            if prev_word != new_word:
                break
            lcp_length += 1

        # On ne confirme jamais la totalité de l'hypothèse en une fois :
        # le dernier mot est quasi-systématiquement encore susceptible
        # d'être révisé par le contexte à venir (mot coupé, homophone,
        # etc.), donc on le laisse toujours dans la partie "partielle" tant
        # que ce n'est pas un Endpoint explicite.
        if lcp_length >= len(new_hypothesis):
            lcp_length = max(0, len(new_hypothesis) - 1)

        result = LocalAgreementResult(
            confirmed_words=new_hypothesis[:lcp_length],
            partial_words=new_hypothesis[lcp_length:],
        )

        self._previous_hypothesis = new_hypothesis
        return result

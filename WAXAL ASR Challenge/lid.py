"""Text language identification (LID) for Lingala / Shona / Luganda.

The Phase 2 audio comes without any language metadata, so the language of each clip is predicted from a first-pass
transcription, using a character n-gram Naive Bayes classifier trained on the WAXAL `train` transcriptions
(99-100 % accuracy on the WAXAL validation split)."""
import math
from collections import Counter

from common import load_split, normalize


class CharNB:
    """Multinomial Naive Bayes over character 1..n-grams of normalised text, with add-0.5 smoothing."""

    def __init__(self, n=4):
        """
        Parameters:
            n (int): longest character n-gram used as a feature.
        """
        self.n, self.counts, self.tot, self.prior = n, {}, {}, {}

    def _grams(self, text):
        """Return every character 1..n-gram of the normalised text, padded with one space on each side."""
        padded = f" {normalize(text)} "
        return [padded[i:i + k] for k in range(1, self.n + 1) for i in range(len(padded) - k + 1)]

    def fit(self, texts, langs):
        """Count n-grams per language.

        Parameters:
            texts (list[str]): training transcriptions.
            langs (list[str]): language code of each transcription.

        Returns:
            CharNB: the fitted classifier (self).
        """
        for lang in set(langs):
            self.counts[lang] = Counter()
        for text, lang in zip(texts, langs):
            self.counts[lang].update(self._grams(text))
        self.V = len(set().union(*self.counts.values()))
        self.tot = {lang: sum(c.values()) for lang, c in self.counts.items()}
        n_per_lang = Counter(langs)
        self.prior = {lang: math.log(n_per_lang[lang] / len(langs)) for lang in n_per_lang}
        return self

    def predict(self, text):
        """
        Parameters:
            text (str): a transcription (any casing / punctuation).

        Returns:
            str: the most likely language code.
        """
        grams = self._grams(text)
        log_post = {lang: self.prior[lang] + sum(math.log((self.counts[lang][g] + 0.5) / (self.tot[lang] + 0.5 * self.V))
                                                 for g in grams)
                    for lang in self.counts}
        return max(log_post, key=log_post.get)


def train_lid():
    """Fit the LID classifier on all WAXAL `train` transcriptions (lin, sna, lug).

    Returns:
        CharNB: fitted classifier; `predict(text)` returns 'lin', 'sna' or 'lug'.
    """
    train = load_split("train")
    return CharNB().fit(train.text.fillna("").tolist(), train.language.tolist())
